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
        self.linearXIn = ceildiv(ceildiv(ceildiv(ceildiv(input_shape[0], 2), 2), 2), 2) # needs to match stride each layer
        self.linearYIn = ceildiv(ceildiv(ceildiv(ceildiv(input_shape[1], 2), 2), 2), 2)
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
            nn.Linear(256 * self.linearXIn * self.linearYIn, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256 * self.linearXIn * self.linearYIn),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, self.linearXIn, self.linearYIn)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = F.interpolate(decoded, size=self.input_shape, mode='bilinear', align_corners=False)
        return decoded

    def encode(self, x):
        return self.encoder(x)

class EnvironmentAutoencoder:
    def __init__(self, input_shape, device):
        self.device = device
        print(f"Autoencoder using device: {self.device}")
        self.input_shape = input_shape  # (4, X, Y)
        
        # Create 3 autoencoders
        self.autoencoders = [
            LayerAutoencoder((input_shape[1], input_shape[2])),  # For layer 0
            LayerAutoencoder((input_shape[1], input_shape[2])),  # For layers 1 and 2
            LayerAutoencoder((input_shape[1], input_shape[2]))   # For layer 3
        ]
        
        self.optimizers = [optim.Adam(ae.parameters(), lr=0.0001) for ae in self.autoencoders]
        self.scaler = amp.GradScaler()
        
        self.scalers = [
            lambda x: x,  # For binary layer (layer 0)
            lambda x: (x + 20) / 20,  # For negative layers (layers 1 and 2)
            lambda x: (x + 20) / 20,  # For negative layers (layers 1 and 2)
            lambda x: (x + 20) / 20   # For negative layer (layer 3)
        ]
        
        self.inverse_scalers = [
            lambda x: x,  # For binary layer (layer 0)
            lambda x: x * 20 - 20,  # For negative layers (layers 1 and 2)
            lambda x: x * 20 - 20,  # For negative layers (layers 1 and 2)
            lambda x: x * 20 - 20   # For negative layer (layer 3)
        ]

    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Binary case (0/1)
            return F.mse_loss(recon_x, x, reduction='mean')
        else:  # -20 to 0 case (including jammer layer)
            x_rescaled = self.inverse_scalers[layer](x)
            recon_x_rescaled = self.inverse_scalers[layer](recon_x)
            
            # Create a mask for values above -20
            mask = (x_rescaled > -20).float()
            
            # Calculate the proportion of values above -20
            proportion_above_threshold = mask.mean()
            
            # Calculate the weight for values above -20
            # The smaller the proportion, the higher the weight
            weight_above_threshold = 1 / (proportion_above_threshold + 1e-6)
            
            # Create a weight tensor
            weights = torch.ones_like(x_rescaled)
            weights = torch.where(mask == 1, weight_above_threshold, weights)
            
            # Calculate weighted MSE loss
            mse_loss = torch.mean(weights * (recon_x_rescaled - x_rescaled)**2)
            
            return mse_loss

    def train_step(self, batch, layer):
        ae = self.autoencoders[layer].to(self.device)
        optimizer = self.optimizers[layer]
        ae.train()

        layer_input = self.scalers[layer](batch[f'layer_{layer}']).to(self.device)
        
        with amp.autocast():
            outputs = ae(layer_input.unsqueeze(1))  # Add channel dimension
            loss = self.custom_loss(outputs, layer_input.unsqueeze(1), layer)

            # Check for nan loss
            if torch.isnan(loss):
                print(f"NaN loss detected in layer {layer}")
                print(f"Input min: {layer_input.min()}, max: {layer_input.max()}")
                print(f"Output min: {outputs.min()}, max: {outputs.max()}")
                return None, None, None

        # Compute gradients
        self.scaler.scale(loss).backward()
        
        # Unscale gradients for logging and clipping
        self.scaler.unscale_(optimizer)
        
        # Compute gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
        
        # Update weights
        self.scaler.step(optimizer)
        self.scaler.update()
        
        # Compute weight update norm
        with torch.no_grad():
            total_update_norm = sum((p.data - p.old_data).norm(2).item() ** 2 for p in ae.parameters() if hasattr(p, 'old_data')) ** 0.5
            for p in ae.parameters():
                p.old_data = p.data.clone()
        
        optimizer.zero_grad()
        
        return loss.item(), total_norm.item(), total_update_norm

    def move_to_cpu(self, layer):
        self.autoencoders[layer] = self.autoencoders[layer].to('cpu')
        torch.cuda.empty_cache()

    def encode_state(self, state):
        with torch.no_grad():
            scaled_state = [self.scalers[i](state[i]) for i in range(len(state))]
            state_tensor = torch.FloatTensor(scaled_state).unsqueeze(1)  # Add channel dimension
            encoded_layers = []
            for i in range(4):  # We have 4 layers in the state
                if i == 0:
                    ae_index = 0  # Use first autoencoder for map layer
                elif i in [1, 2]:
                    ae_index = 1  # Use second autoencoder for agent and target layers
                else:
                    ae_index = 2  # Use third autoencoder for jammer layer
                ae = self.autoencoders[ae_index]
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
            for i in range(4):  # We have 4 layers in the state
                if i == 0:
                    ae_index = 0  # Use first autoencoder for map layer
                elif i in [1, 2]:
                    ae_index = 1  # Use second autoencoder for agent and target layers
                else:
                    ae_index = 2  # Use third autoencoder for jammer layer
                ae = self.autoencoders[ae_index]
                ae.eval()
                ae.to(self.device)
                encoded_layer = torch.FloatTensor(encoded_state[i]).unsqueeze(0).unsqueeze(0).to(self.device)
                decoded_layer = ae.decoder(encoded_layer)
                decoded_layer = self.inverse_scalers[ae_index](decoded_layer.cpu().numpy().squeeze())
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