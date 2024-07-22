import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import numpy as np

class LayerAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(LayerAutoencoder, self).__init__()
        self.input_shape = input_shape  # (X, Y)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * (input_shape[0] // 8) * (input_shape[1] // 8), 256)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 128 * (input_shape[0] // 8) * (input_shape[1] // 8)),
            nn.Unflatten(1, (128, input_shape[0] // 8, input_shape[1] // 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

class EnvironmentAutoencoder:
    def __init__(self, input_shape, device):
        self.device = device
        print(f"Autoencoder using device: {self.device}")
        self.input_shape = input_shape  # (D, X, Y)
        self.autoencoders = nn.ModuleList([LayerAutoencoder((input_shape[1], input_shape[2])).to(device) for _ in range(input_shape[0])])
        self.optimizers = [optim.Adam(ae.parameters(), lr=0.001) for ae in self.autoencoders]
        self.criterion = nn.MSELoss()
        self.scalers = [amp.GradScaler() for _ in range(input_shape[0])]  # For mixed precision training

    def train(self, dataloader):
        for ae in self.autoencoders:
            ae.train()
        
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            
            loss = 0
            for i, ae in enumerate(self.autoencoders):
                self.optimizers[i].zero_grad()
                layer_input = batch[:, i:i+1, :, :]  # Select one layer and keep dimension
                
                # Use mixed precision training
                with amp.autocast():
                    outputs = ae(layer_input)
                    layer_loss = self.criterion(outputs, layer_input)
                
                # Scale the loss and call backward
                self.scalers[i].scale(layer_loss).backward()
                self.scalers[i].step(self.optimizers[i])
                self.scalers[i].update()
                
                loss += layer_loss.item()
            
            total_loss += loss / len(self.autoencoders)
        
        return total_loss / len(dataloader)

    def encode_state(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(1).to(self.device)  # Add channel dimension
            encoded_layers = []
            for i, ae in enumerate(self.autoencoders):
                ae.eval()
                layer_input = state_tensor[:, :, i, :, :]  # Select one layer
                encoded_layer = ae.encode(layer_input)
                encoded_layers.append(encoded_layer.cpu().numpy().squeeze())
        return np.array(encoded_layers)

    def save(self, path):
        torch.save({
            'model_state_dicts': [ae.state_dict() for ae in self.autoencoders],
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'input_shape': self.input_shape
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for i, ae in enumerate(self.autoencoders):
            ae.load_state_dict(checkpoint['model_state_dicts'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dicts'][i])
            ae.eval()