import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

class LayerAutoencoder(nn.Module):
    def __init__(self, is_map=False):
        super(LayerAutoencoder, self).__init__()
        self.is_map = is_map

        # Calculate the flattened size
        self.flattened_size = 256 * 16 * 8  # 32,768
        # Encoder convolutional layers (no padding)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(self.flattened_size, 8192),
            nn.LeakyReLU(0.2),
            nn.Linear(8192, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128)
        )

        # Decoder linear layers with gradual expansion
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 8192),
            nn.LeakyReLU(0.2),
            nn.Linear(8192, self.flattened_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 16, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        # Decoder
        decoded = self.decoder(encoded)
        # Ensure output size is 276x155 using interpolation
        decoded = F.interpolate(decoded, size=(276, 155), mode='bilinear', align_corners=False)

        torch.cuda.empty_cache()
        
        if not self.is_map:
            # Apply thresholding to mitigate interpolation effects for continuous data
            background_mask = (decoded <= -19.9).float()
            decoded = decoded * (1 - background_mask) + (-20) * background_mask
            decoded = torch.clamp(x, min=-20, max=0) #TODO: modify this to scale to -20-0 rather than clamp?
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        decoded = self.decoder(x)
        return F.interpolate(decoded, size=(276, 155), mode='bilinear', align_corners=False)

class EnvironmentAutoencoder:
    def __init__(self, device):
        self.device = device
        print(f"Autoencoder using device: {self.device}")
        
        self.autoencoders = [
            LayerAutoencoder(is_map=True),  # For layer 0 (map)
            LayerAutoencoder(),  # For layers 1 and 2
            LayerAutoencoder()   # For layer 3
        ]
        
        self.optimizers = [torch.optim.Adam(ae.parameters(), lr=0.0001) for ae in self.autoencoders]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5) for opt in self.optimizers]
        self.scaler = GradScaler()

    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Binary case (0/1)
            return F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
            # return F.mse_loss(recon_x, x, reduction='mean')
        else:  # -20 to 0 case (including jammer layer)
            # Create masks for background and non-background values
            background_mask = (x == -20).float()
            nonbackground_mask = (x > -20).float()
            # Calculate the proportion of non-background values
            proportion_nonbackground = nonbackground_mask.mean()
            # Set a minimum proportion to avoid division by zero
            min_proportion = 1e-6
            proportion_nonbackground = max(proportion_nonbackground, min_proportion)
            # Calculate weights for background and non-background
            background_weight = 1  # Small weight for background
            nonbackground_weight = 1 / proportion_nonbackground  # Higher weight for non-background
            # Compute MSE for background and non-background separately
            background_mse = F.mse_loss(recon_x * background_mask, x * background_mask, reduction='sum')
            nonbackground_mse = F.mse_loss(recon_x * nonbackground_mask, x * nonbackground_mask, reduction='sum')
            # Compute L1 loss for non-background to encourage sparsity and exact reconstruction
            nonbackground_l1 = F.l1_loss(recon_x * nonbackground_mask, x * nonbackground_mask, reduction='sum')
            # Combine losses with appropriate weights
            total_loss = (background_weight * background_mse +
                        nonbackground_weight * (nonbackground_mse + 10 * nonbackground_l1)) / x.numel()
            return total_loss

    def train_step(self, batch, layer):
        ae = self.autoencoders[layer]
        optimizer = self.optimizers[layer]
        ae.train()

        layer_input = batch.to(self.device)
        
        optimizer.zero_grad()
        
        with autocast():
            layer_input = layer_input.unsqueeze(1)
            outputs = ae(layer_input) # Add channel dim
            
            loss = self.custom_loss(outputs, layer_input, layer) # remove channel dim, calc loss
        
        if torch.isnan(loss):
            print(f"NaN loss detected in layer {layer}")
            print(f"Input shape: {layer_input.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Input min: {layer_input.min()}, max: {layer_input.max()}")
            print(f"Output min: {outputs.min()}, max: {outputs.max()}")
            return None
        
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        loss_value = loss.item()
        self.schedulers[layer].step(loss_value)
        torch.cuda.empty_cache()
        return loss_value
    
    def move_to_cpu(self, layer):
        self.autoencoders[layer] = self.autoencoders[layer].to('cpu')
        torch.cuda.empty_cache()

    def encode_state(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(1)  # Add channel dimension
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
                decoded_layers.append(decoded_layer.cpu().numpy().squeeze())
                ae.to('cpu')
                torch.cuda.empty_cache()
        return np.array(decoded_layers)
    
    def save(self, path):
        torch.save({
            'model_state_dicts': [ae.cpu().state_dict() for ae in self.autoencoders],
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'scheduler_state_dicts': [sch.state_dict() for sch in self.schedulers],
            'scaler': self.scaler.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        for i, ae in enumerate(self.autoencoders):
            ae.load_state_dict(checkpoint['model_state_dicts'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dicts'][i])
            self.schedulers[i].load_state_dict(checkpoint['scheduler_state_dicts'][i])
        self.scaler.load_state_dict(checkpoint['scaler'])