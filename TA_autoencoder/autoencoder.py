import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np


class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SparseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.mask = nn.Parameter(torch.ones_like(self.conv.weight))
        
    def forward(self, x):
        sparse_weight = self.conv.weight * self.mask
        return F.conv2d(x, sparse_weight.to(x.dtype), self.conv.bias.to(x.dtype), self.conv.stride, self.conv.padding)
    
    def get_mask(self):
        return self.mask

class CustomFinalUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, target_size):
        super(CustomFinalUpsampling, self).__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Use nearest neighbor interpolation for initial upsampling
        x = F.interpolate(x, size=self.target_size, mode='nearest')
        # Apply 1x1 convolution for final adjustments
        return self.conv(x.to(self.conv.weight.dtype))

class LayerAutoencoder(nn.Module):
    def __init__(self, is_map=False):
        super(LayerAutoencoder, self).__init__()
        self.is_map = is_map

        # Calculate the flattened size
        self.flattened_size = 256 * 7 * 3  # 5376
        # Encoder
        self.encoder = nn.Sequential(
            SparseConv2d(1, 16, kernel_size=3, stride=2), # 138, 77
            nn.LeakyReLU(0.2),
            SparseConv2d(16, 32, kernel_size=3, stride=2), # 69, 38
            nn.LeakyReLU(0.2),
            SparseConv2d(32, 64, kernel_size=3, stride=2), # 34, 19
            nn.LeakyReLU(0.2),
            SparseConv2d(64, 128, kernel_size=3, stride=2), # 17, 9
            nn.LeakyReLU(0.2),
            SparseConv2d(128, 256, kernel_size=3, stride=2), # 7, 3 # Unsure why it doesn't go to 8, 4
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(self.flattened_size, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, self.flattened_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 7, 3)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1),
            nn.LeakyReLU(0.2),
            CustomFinalUpsampling(16, 1, (276, 155))
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
        
        if not self.is_map:
            background_mask = (decoded <= -19.9).float()
            decoded = decoded * (1 - background_mask) + (-20) * background_mask
            decoded = torch.clamp(decoded, min=-20, max=0)
        
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        decoded = self.decoder(x)
        return F.interpolate(decoded, size=(276, 155), mode='bilinear', align_corners=False)
    
    def get_sparse_layers(self):
        return [layer for layer in self.encoder if isinstance(layer, SparseConv2d)]

class EnvironmentAutoencoder:
    def __init__(self, device, initial_l1_weight=1e-5, max_grad_norm=1.0, mask_regularisation_weight=1e-4):
        self.device = device
        print(f"Autoencoder using device: {self.device}")
        
        self.autoencoders = [
            LayerAutoencoder(is_map=True),  # For layer 0 (map)
            LayerAutoencoder(),  # For layers 1 and 2
            LayerAutoencoder()   # For layer 3
        ]
        
        self.optimizers = [torch.optim.Adam(ae.parameters(), lr=0.0001, weight_decay=1e-5) for ae in self.autoencoders]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5) for opt in self.optimizers]
        self.scaler = GradScaler()
        
        self.l1_weight = initial_l1_weight
        self.max_grad_norm = max_grad_norm
        self.mask_regularisation_weight = mask_regularisation_weight
        self.l1_history = []
        self.reconstruction_loss_history = []
        self.mask_regularisation_history = []

    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Binary case (0/1)
            reconstruction_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
        else:  # -20 to 0 case (including jammer layer)
            background_mask = (x == -20).float()
            nonbackground_mask = (x > -20).float()
            proportion_nonbackground = max(nonbackground_mask.mean(), 1e-6)
            
            background_weight = 1
            nonbackground_weight = 1 / proportion_nonbackground
            
            background_mse = F.mse_loss(recon_x * background_mask, x * background_mask, reduction='sum')
            nonbackground_mse = F.mse_loss(recon_x * nonbackground_mask, x * nonbackground_mask, reduction='sum')
            nonbackground_l1 = F.l1_loss(recon_x * nonbackground_mask, x * nonbackground_mask, reduction='sum')
            
            reconstruction_loss = (background_weight * background_mse +
                        nonbackground_weight * (nonbackground_mse + 10 * nonbackground_l1)) / x.numel()
        
        # Add L1 regularisation for sparsity
        l1_reg = sum(p.abs().sum() for p in self.autoencoders[layer].parameters())
        
        # Add mask regularisation
        mask_reg = sum(layer.get_mask().abs().sum() for layer in self.autoencoders[layer].get_sparse_layers())
        
        total_loss = reconstruction_loss + self.l1_weight * l1_reg + self.mask_regularisation_weight * mask_reg
        
        return total_loss, reconstruction_loss, l1_reg, mask_reg

    def train_step(self, batch, layer):
        ae = self.autoencoders[layer]
        optimizer = self.optimizers[layer]
        ae.train()

        layer_input = batch.to(self.device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            layer_input = layer_input.unsqueeze(1)
            layer_input = layer_input.to(torch.float32) # Ensure input is float32 before processing
            outputs = ae(layer_input)
            loss, reconstruction_loss, l1_reg, mask_reg = self.custom_loss(outputs, layer_input, layer)
        
        if torch.isnan(loss):
            print(f"NaN loss detected in layer {layer}")
            print(f"Input shape: {layer_input.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Input min: {layer_input.min()}, max: {layer_input.max()}")
            print(f"Output min: {outputs.min()}, max: {outputs.max()}")
            return None
        
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(ae.parameters(), self.max_grad_norm)
        
        self.scaler.step(optimizer)
        self.scaler.update()
        
        loss_value = loss.item()
        self.schedulers[layer].step(loss_value)
        
        # Record losses for monitoring
        self.reconstruction_loss_history.append(reconstruction_loss.item())
        self.l1_history.append(l1_reg.item())
        self.mask_regularisation_history.append(mask_reg.item())
        
        del outputs
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        return loss_value

    def adjust_regularisation_weights(self, window_size=100):
        if len(self.reconstruction_loss_history) < window_size:
            return
        
        recent_reconstruction = np.mean(self.reconstruction_loss_history[-window_size:])
        recent_l1 = np.mean(self.l1_history[-window_size:])
        recent_mask_reg = np.mean(self.mask_regularisation_history[-window_size:])
        
        # Adjust L1 weight
        ratio_l1 = recent_reconstruction / (recent_l1 + 1e-10)
        if ratio_l1 > 100:
            self.l1_weight *= 2
        elif ratio_l1 < 10:
            self.l1_weight /= 2
        
        # Adjust mask regularisation weight
        ratio_mask = recent_reconstruction / (recent_mask_reg + 1e-10)
        if ratio_mask > 100:
            self.mask_regularisation_weight *= 2
        elif ratio_mask < 10:
            self.mask_regularisation_weight /= 2
        
        # print(f"Adjusted L1 weight: {self.l1_weight}, Mask regularisation weight: {self.mask_regularisation_weight}")

    def train_epoch(self, dataloader, layer):
        total_loss = 0
        num_batches = 0
        for batch in dataloader:
            loss = self.train_step(batch, layer)
            if loss is not None:
                total_loss += loss
                num_batches += 1
            
            if num_batches % 100 == 0:
                self.adjust_l1_weight()
        
        return total_loss / num_batches if num_batches > 0 else None
    
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