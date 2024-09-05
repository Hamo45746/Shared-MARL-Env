import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os

def ceildiv(a, b):
    return -(a // -b)

# class SpatialAttention2D(nn.Module):
#     def __init__(self, in_channels):
#         super(SpatialAttention2D, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
#         self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=3, padding=1)
        
#     def forward(self, x):
#         attn = self.conv1(x)
#         attn = F.relu(attn)
#         attn = self.conv2(attn)
#         attn = torch.sigmoid(attn)
#         return x * attn

# class SparseConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1):
#         super(SparseConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
#         self.mask = nn.Parameter(torch.ones_like(self.conv.weight))
#         self.attention = SpatialAttention2D(out_channels)
        
#     def forward(self, x):
#         sparse_weight = self.conv.weight * self.mask
#         x = F.conv2d(x, sparse_weight, self.conv.bias, self.conv.stride, self.conv.padding)
#         x = self.attention(x)
#         return x
    
#     def get_mask(self):
#         return self.mask

# class CustomFinalUpsampling(nn.Module):
#     def __init__(self, in_channels, out_channels, target_size):
#         super(CustomFinalUpsampling, self).__init__()
#         self.target_size = target_size
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
#     def forward(self, x):
#         # Use nearest neighbor interpolation for initial upsampling
#         x = F.interpolate(x, size=self.target_size, mode='nearest')
#         # Apply 1x1 convolution for final adjustments
#         return self.conv(x.to(self.conv.weight.dtype))


class LayerAutoencoder(nn.Module):
    def __init__(self, is_map=False):
        super(LayerAutoencoder, self).__init__()
        self.is_map = is_map
        self.latent_dim = 256
        self.input_shape = (276, 155)

        # Encoder
        # Input: 1 x 276 x 155
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Input: 64 x 138 x 78
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Input: 128 x 69 x 39
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Input: 256 x 35 x 20
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Input: 512 x 18 x 10
        self.flatten = nn.Flatten()
        self.fc_encoder = nn.Linear(512 * 18 * 10, self.latent_dim)

        # Decoder
        self.fc_decoder = nn.Linear(self.latent_dim, 512 * 18 * 10)
        # Input: 512 x 18 x 10
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Input: 512 x 35 x 20 (including skip connection)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Input: 256 x 69 x 39 (including skip connection)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        # Input: 128 x 138 x 78 (including skip connection)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        )
        # Output: 1 x 276 x 155

    def forward(self, x):
        # Encoding
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Latent space
        flattened = self.flatten(e4)
        latent = self.fc_encoder(flattened)

        # Decoding
        d = self.fc_decoder(latent)
        d = d.view(-1, 512, 18, 10)
        d = self.dec4(d)
        d = self.dec3(torch.cat([d, e3], dim=1))
        d = self.dec2(torch.cat([d, e2], dim=1))
        decoded = self.dec1(torch.cat([d, e1], dim=1))

        # Post-processing
        if not self.is_map:
            decoded = -20 + 20 * torch.sigmoid(decoded)
            background_mask = (decoded <= -19.8).float()
            decoded = decoded * (1 - background_mask) + (-20) * background_mask
        else:
            decoded = torch.sigmoid(decoded)

        return decoded

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        flattened = self.flatten(e4)
        return self.fc_encoder(flattened)

    def decode(self, latent):
        d = self.fc_decoder(latent)
        d = d.view(-1, 512, 18, 10)
        d = self.dec4(d)
        d = self.dec3(torch.cat([d, torch.zeros_like(d)], dim=1))
        d = self.dec2(torch.cat([d, torch.zeros_like(d)], dim=1))
        decoded = self.dec1(torch.cat([d, torch.zeros_like(d)], dim=1))
        
        # Post-processing
        if not self.is_map:
            decoded = -20 + 20 * torch.sigmoid(decoded)
            background_mask = (decoded <= -19.8).float()
            decoded = decoded * (1 - background_mask) + (-20) * background_mask
        else:
            decoded = torch.sigmoid(decoded)
        return decoded


class EnvironmentAutoencoder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        print(f"Autoencoder using device: {self.device}")
        
        self.autoencoders = [
            LayerAutoencoder(is_map=True),  # For layer 0 (map)
            LayerAutoencoder(),  # For layers 1 and 2
            LayerAutoencoder()   # For layer 3
        ]
        
        self.optimizers = [torch.optim.Adam(ae.parameters(), lr=0.0001) for ae in self.autoencoders]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.4, patience=8, min_lr=1e-5) for opt in self.optimizers]
        self.scaler = GradScaler()
        
        # self.scalers = [
        #     lambda x: x,  # For binary layer (layer 0)
        #     lambda x: (x + 20) / 20,  # For negative layers (layers 1 and 2)
        #     lambda x: (x + 20) / 20,  # For negative layers (layers 1 and 2)
        #     lambda x: (x + 20) / 20   # For negative layer (layer 3)
        # ]
        
        # self.inverse_scalers = [
        #     lambda x: x,  # For binary layer (layer 0)
        #     lambda x: x * 20 - 20,  # For negative layers (layers 1 and 2)
        #     lambda x: x * 20 - 20,  # For negative layers (layers 1 and 2)
        #     lambda x: x * 20 - 20   # For negative layer (layer 3)
        # ]


    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Binary case (0/1)
            return F.mse_loss(recon_x, x, reduction='mean')
        else:  # -20 to 0 case (including jammer layer)
            # x_rescaled = self.inverse_scalers[layer](x)
            # recon_x_rescaled = self.inverse_scalers[layer](recon_x)
            x_rescaled = x
            recon_x_rescaled = recon_x
            
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
        ae = self.autoencoders[layer]
        optimizer = self.optimizers[layer]
        ae.train()

        layer_input = batch.to(self.device, dtype=self.dtype)
        layer_input = layer_input.unsqueeze(1)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = ae(layer_input)
            loss = self.custom_loss(outputs, layer_input, layer)
        
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
        
        del outputs
        optimizer.zero_grad(set_to_none=True)
        return loss_value

    def move_to_cpu(self, layer):
        self.autoencoders[layer] = self.autoencoders[layer].cpu()
        torch.cuda.empty_cache()

    def encode_state(self, state):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=self.dtype).unsqueeze(1)  # Add channel dimension
            encoded_layers = []
            for i in range(4):  # We have 4 layers in the state
                if i == 0:
                    ae_index = 0  # Use first autoencoder for map layer
                elif i in [1, 2]:
                    ae_index = 1  # Use second autoencoder for agent and target layers
                else:
                    ae_index = 2  # Use third autoencoder for jammer layer
                ae = self.autoencoders[ae_index]
                ae.eval()  # Ensure it's in eval mode
                ae.to(self.device)
                layer_input = state_tensor[i].unsqueeze(0).to(self.device, dtype=self.dtype)  # Add batch dimension and move to GPU
                encoded_layer = ae.encode(layer_input)
                encoded_layers.append(encoded_layer.cpu().numpy().squeeze())
                ae.cpu()
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
                ae.eval()  # Ensure it's in eval mode
                ae.to(self.device)
                encoded_layer = torch.tensor(encoded_state[i], dtype=self.dtype).unsqueeze(0).unsqueeze(0).to(self.device)
                decoded_layer = ae.decode(encoded_layer)
                decoded_layers.append(decoded_layer.cpu().numpy().squeeze())
                ae.cpu()
                torch.cuda.empty_cache()
            return np.array(decoded_layers)


    def save(self, path):
        torch.save({
            'model_state_dicts': [ae.cpu().state_dict() for ae in self.autoencoders],
            'optimizer_state_dicts': [self.cpu_state_dict(opt) for opt in self.optimizers],
            'scheduler_state_dicts': [sch.state_dict() for sch in self.schedulers],
            'scaler': self.scaler.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        for i, ae in enumerate(self.autoencoders):
            ae.load_state_dict(checkpoint['model_state_dicts'][i])
            ae.to(self.device, dtype=self.dtype)
        
        # for i, opt in enumerate(self.optimizers):
        #     opt.load_state_dict(checkpoint['optimizer_state_dicts'][i])
        #     self.move_optimizer_to_device(opt, self.device)
        
        # for i, sch in enumerate(self.schedulers):
        #     sch.load_state_dict(checkpoint['scheduler_state_dicts'][i])
        
        self.scaler.load_state_dict(checkpoint['scaler'])
        
    def save_single_autoencoder(self, path, index):
        if index not in [0, 1, 2]:
            raise ValueError("Index must be 0, 1, or 2")
        torch.save({
            'model_state_dict': self.autoencoders[index].cpu().state_dict(),
            'is_map': self.autoencoders[index].is_map,
        }, path)

    def load_single_autoencoder(self, path, index):
        if index not in [0, 1, 2]:
            raise ValueError("Index must be 0, 1, or 2")

        checkpoint = torch.load(path, map_location='cpu')
        is_map = checkpoint['is_map']
        
        self.autoencoders[index] = LayerAutoencoder(is_map=is_map)
        self.autoencoders[index].load_state_dict(checkpoint['model_state_dict'])
        self.autoencoders[index].to(self.device)
        
        # Reinitialize optimizer and scheduler
        self.optimizers[index] = torch.optim.Adam(self.autoencoders[index].parameters(), lr=0.0001)
        self.schedulers[index] = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizers[index], mode='min', factor=0.4, patience=8, min_lr=1e-5
        )

    def load_all_autoencoders(self, folder_path):
        """
        Load all three autoencoders from individual save files.
        
        :param folder_path: Path to the folder containing the autoencoder save files
        """
        for index in range(3):
            file_path = os.path.join(folder_path, f"autoencoder_{index}_best.pth")
            if os.path.exists(file_path):
                self.load_single_autoencoder(file_path, index)

    
    @staticmethod
    def cpu_state_dict(optimizer):
        return {k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in optimizer.state_dict().items()}

    @staticmethod
    def move_optimizer_to_device(optimizer, device):
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)