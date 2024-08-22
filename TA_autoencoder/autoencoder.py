import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np

def ceildiv(a, b):
    return -(a // -b)

class SpatialAttention2D(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        attn = self.conv1(x)
        attn = F.relu(attn)
        attn = self.conv2(attn)
        attn = torch.sigmoid(attn)
        return x * attn

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(SparseConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.mask = nn.Parameter(torch.ones_like(self.conv.weight))
        self.attention = SpatialAttention2D(out_channels)
        
    def forward(self, x):
        sparse_weight = self.conv.weight * self.mask
        x = F.conv2d(x, sparse_weight, self.conv.bias, self.conv.stride, self.conv.padding)
        x = self.attention(x)
        return x
    
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
        self.flattened_size = 256 * 9 * 5  # 5376
        input_shape = (276, 155)
        self.input_shape = input_shape  # (276, 155)
        self.linearXIn = ceildiv(ceildiv(ceildiv(ceildiv(input_shape[1], 2), 2), 2), 2) # needs to match stride each layer
        self.linearYIn = ceildiv(ceildiv(ceildiv(ceildiv(input_shape[0], 2), 2), 2), 2)
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
            nn.Unflatten(1, (256, self.linearYIn, self.linearXIn)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # CustomFinalUpsampling(32, 1, (276, 155))
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)
        )

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        
        if not self.is_map:
            # Scale the output to be between -20 and 0
            decoded = -20 + 20 * torch.sigmoid(decoded)
            # Apply background mask
            background_mask = (decoded <= -19.8).float()
            decoded = decoded * (1 - background_mask) + (-20) * background_mask
        else:
            # For the map layer, we keep the binary output
            decoded = torch.sigmoid(decoded)
        decoded = F.interpolate(decoded, size=(276, 155), mode='bilinear', align_corners=False)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        decoded = self.decoder(x)
        return F.interpolate(decoded, size=(276, 155), mode='bilinear', align_corners=False)
    
    def get_sparse_layers(self):
        return [layer for layer in self.encoder if isinstance(layer, SparseConv2d)]

class EnvironmentAutoencoder:
    def __init__(self, device):
        self.device = device
        self.dtype = torch.float32
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
        # layer_input = self.scalers[layer](batch.to(self.device, dtype=self.dtype))
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            layer_input = layer_input.unsqueeze(1)
            layer_input = layer_input.to(self.device, dtype=self.dtype) # Ensure input is float32 before processing
            outputs = ae(layer_input)
            # loss, reconstruction_loss, l1_reg, mask_reg = self.custom_loss(outputs, layer_input, layer)
            loss = self.custom_loss(outputs, layer_input, layer)
        
        if torch.isnan(loss):
            print(f"NaN loss detected in layer {layer}")
            print(f"Input shape: {layer_input.shape}")
            print(f"Output shape: {outputs.shape}")
            print(f"Input min: {layer_input.min()}, max: {layer_input.max()}")
            print(f"Output min: {outputs.min()}, max: {outputs.max()}")
            return None
        
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        # self.scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(ae.parameters(), self.max_grad_norm)
        
        self.scaler.step(optimizer)
        self.scaler.update()
        
        loss_value = loss.item()
        self.schedulers[layer].step(loss_value)
        
        del outputs
        # torch.cuda.empty_cache()
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
                ae.eval()
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
                ae.eval()
                ae.to(self.device)
                encoded_layer = torch.tensor(encoded_state[i], dtype=self.dtype).unsqueeze(0).unsqueeze(0).to(self.device)
                decoded_layer = ae.decoder(encoded_layer)
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
        
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(checkpoint['optimizer_state_dicts'][i])
            self.move_optimizer_to_device(opt, self.device)
        
        for i, sch in enumerate(self.schedulers):
            sch.load_state_dict(checkpoint['scheduler_state_dicts'][i])
        
        self.scaler.load_state_dict(checkpoint['scaler'])

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