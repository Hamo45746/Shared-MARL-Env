import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
import numpy as np

def ceildiv(a, b):
    return -(a // -b)

class LayerAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(LayerAutoencoder, self).__init__()
        self.input_shape = input_shape  # (276, 155)
        linearXIn = ceildiv(ceildiv(ceildiv(ceildiv(input_shape[0], 2), 2), 2), 2) # needs to match stride each layer
        linearYIn = ceildiv(ceildiv(ceildiv(ceildiv(input_shape[1], 2), 2), 2), 2)
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * linearXIn * linearYIn, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * linearXIn * linearYIn),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, linearXIn, linearYIn)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        encoded = self.encoder(x)
        print(f"Encoded shape: {encoded.shape}")
        decoded = self.decoder(encoded)
        print(f"Decoded shape: {decoded.shape}")
        return decoded

    def encode(self, x):
        return self.encoder(x)

class EnvironmentAutoencoder:
    def __init__(self, input_shape, device, accumulation_steps=4):
        self.device = device
        print(f"Autoencoder using device: {self.device}")
        self.input_shape = input_shape  # (4, X, Y)
        self.autoencoders = [LayerAutoencoder((input_shape[1], input_shape[2])) for _ in range(input_shape[0])]
        self.optimizers = [optim.Adam(ae.parameters(), lr=0.001) for ae in self.autoencoders]
        self.scaler = amp.GradScaler()
        self.accumulation_steps = accumulation_steps
        
        self.scalers = [lambda x: x]  # For binary layer (should be the first layer)
        self.scalers.extend([lambda x: (x + 20) / 20 for _ in range(input_shape[0] - 1)])  # For negative layers
        
        self.inverse_scalers = [lambda x: x]  # For binary layer
        self.inverse_scalers.extend([lambda x: x * 20 - 20 for _ in range(input_shape[0] - 1)])  # For negative layers

    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Binary case (0/1)
            return F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
        else:  # -20 to 0 case
            # Rescale x back to -20 to 0 range for loss calculation
            x_rescaled = self.inverse_scalers[layer](x)
            recon_x_rescaled = self.inverse_scalers[layer](recon_x)
            
            # Weighted MSE loss
            weights = torch.exp(x_rescaled / 20)  # More weight to values closer to 0
            mse_loss = torch.mean(weights * (recon_x_rescaled - x_rescaled)**2)
            
            return mse_loss

    def train_step(self, batch, layer):
        ae = self.autoencoders[layer].to(self.device)
        optimizer = self.optimizers[layer]
        ae.train()

        layer_input = self.scalers[layer](batch[f'layer_{layer}']).to(self.device)
        
        print(f"Layer {layer} input shape: {layer_input.shape}")
        
        with amp.autocast():
            outputs = ae(layer_input.unsqueeze(1))  # Add channel dimension
            loss = self.custom_loss(outputs, layer_input.unsqueeze(1), layer)
            loss = loss / self.accumulation_steps
        
        self.scaler.scale(loss).backward()
        
        if (layer + 1) % self.accumulation_steps == 0:
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        
        return loss.item() * self.accumulation_steps

    def move_to_cpu(self, layer):
        self.autoencoders[layer] = self.autoencoders[layer].to('cpu')
        torch.cuda.empty_cache()

    def encode_state(self, state):
        with torch.no_grad():
            scaled_state = [self.scalers[i](state[i]) for i in range(len(state))]
            state_tensor = torch.FloatTensor(scaled_state).unsqueeze(1)  # Add channel dimension
            encoded_layers = []
            for i, ae in enumerate(self.autoencoders):
                ae.eval()
                ae.to(self.device)
                layer_input = state_tensor[i].unsqueeze(0).to(self.device)  # Add batch dimension and move to GPU
                encoded_layer = ae.encode(layer_input)
                encoded_layers.append(encoded_layer.cpu().numpy().squeeze())
                ae.to('cpu')
                torch.cuda.empty_cache()
        return np.array(encoded_layers)

    def decode_state(self, encoded_state):
        with torch.no_grad():
            decoded_layers = []
            for i, ae in enumerate(self.autoencoders):
                ae.eval()
                ae.to(self.device)
                encoded_layer = torch.FloatTensor(encoded_state[i]).unsqueeze(0).unsqueeze(0).to(self.device)
                decoded_layer = ae.decoder(encoded_layer)
                decoded_layer = self.inverse_scalers[i](decoded_layer.cpu().numpy().squeeze())
                decoded_layers.append(decoded_layer)
                ae.to('cpu')
                torch.cuda.empty_cache()
        return np.array(decoded_layers)

    def save(self, path):
        torch.save({
            'model_state_dicts': [ae.state_dict() for ae in self.autoencoders],
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'input_shape': self.input_shape
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')  # Load to CPU first
        for i, ae in enumerate(self.autoencoders):
            ae.load_state_dict(checkpoint['model_state_dicts'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dicts'][i])
            ae.eval()