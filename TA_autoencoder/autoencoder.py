import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DynamicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.stride = stride

    def forward(self, x):
        h, w = x.shape[2:]
        h_out = (h - self.conv.kernel_size[0]) // self.stride + 1
        w_out = (w - self.conv.kernel_size[1]) // self.stride + 1
        return self.conv(x[:, :, :h_out*self.stride, :w_out*self.stride])

class DynamicConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DynamicConvTranspose2d, self).__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.stride = stride

    def forward(self, x, output_size):
        return self.convt(x, output_size=output_size)

class LayerAutoencoder(nn.Module):
    def __init__(self, is_map=False):
        super(LayerAutoencoder, self).__init__()
        self.is_map = is_map

        # Encoder convolutional layers (no padding)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=0),  # Output: 137x77
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),  # Output: 68x38
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),  # Output: 33x18
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),  # Output: 16x8
            nn.LeakyReLU(0.2),
        )

        # Calculate the flattened size
        self.flattened_size = 256 * 16 * 8

        # Encoder linear layers
        self.encoder_linear = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64)
        )

        # Decoder linear layers
        self.decoder_linear = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.flattened_size),
            nn.LeakyReLU(0.2)
        )

        # Decoder convolutional layers (no padding)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=0),  # Output: 33x17
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0),  # Output: 67x35
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),  # Output: 135x71
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=0),  # Output: 271x143
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Ensure input is 4D: [batch_size, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        # Encoder
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        encoded = self.encoder_linear(x)

        # Decoder
        x = self.decoder_linear(encoded)
        x = x.view(x.size(0), 256, 16, 8)
        x = self.decoder_conv(x)

        # Ensure output size is 276x155 using interpolation
        x = F.interpolate(x, size=(276, 155), mode='bilinear', align_corners=False)

        if self.is_map:
            return torch.sigmoid(x)
        else:
            return torch.clamp(x, min=-20, max=0)

    def encode(self, x):
        # Ensure input is 4D: [batch_size, channels, height, width]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        elif x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        return self.encoder_linear(x)

class EnvironmentAutoencoder:
    def __init__(self, device):
        self.device = device
        print(f"Autoencoder using device: {self.device}")
        
        self.autoencoders = [
            LayerAutoencoder(is_map=True).to(device),  # For layer 0 (map)
            LayerAutoencoder().to(device),  # For layers 1 and 2
            LayerAutoencoder().to(device)   # For layer 3
        ]
        
        self.optimizers = [torch.optim.Adam(ae.parameters(), lr=0.0001) for ae in self.autoencoders]
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True) for opt in self.optimizers]
        self.scaler = GradScaler()

    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Map layer (Binary Cross Entropy loss)
            return F.binary_cross_entropy(recon_x, x, reduction='mean')
        else:  # Other layers (-20 to 0 range)
            # MSE loss for non-background pixels, L1 loss for background
            mse_loss = F.mse_loss(recon_x, x, reduction='none')
            l1_loss = F.l1_loss(recon_x, x, reduction='none')
            background_mask = (x == -20).float()
            combined_loss = background_mask * l1_loss + (1 - background_mask) * mse_loss
            return combined_loss.mean()

    def train_step(self, batch, layer):
        ae = self.autoencoders[layer]
        optimizer = self.optimizers[layer]
        ae.train()

        layer_input = batch.to(self.device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = ae(layer_input)  # Remove unsqueeze(1) as it's handled in the forward method
            loss = self.custom_loss(outputs.squeeze(1), layer_input, layer)
        
        if torch.isnan(loss):
            print(f"NaN loss detected in layer {layer}")
            print(f"Input min: {layer_input.min()}, max: {layer_input.max()}")
            print(f"Output min: {outputs.min()}, max: {outputs.max()}")
            return None
        
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        self.schedulers[layer].step(loss.item())
        
        return loss.item()

    def encode_state(self, state):
        with torch.no_grad():
            encoded_layers = []
            for i, ae in enumerate(self.autoencoders):
                ae.eval()
                layer_input = torch.FloatTensor(state[i]).unsqueeze(0).unsqueeze(0).to(self.device)
                encoded_layer = ae.encode(layer_input)
                encoded_layers.append(encoded_layer.cpu().numpy().squeeze())
        return encoded_layers

    def decode_state(self, encoded_state):
        with torch.no_grad():
            decoded_layers = []
            for i, ae in enumerate(self.autoencoders):
                ae.eval()
                encoded_layer = torch.FloatTensor(encoded_state[i]).unsqueeze(0).to(self.device)
                decoded_layer = ae(encoded_layer.unsqueeze(2).unsqueeze(3))
                decoded_layers.append(decoded_layer.cpu().numpy().squeeze())
        return decoded_layers

    def save(self, path):
        torch.save({
            'model_state_dicts': [ae.state_dict() for ae in self.autoencoders],
            'optimizer_state_dicts': [opt.state_dict() for opt in self.optimizers],
            'scheduler_state_dicts': [sch.state_dict() for sch in self.schedulers],
            'scaler': self.scaler.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        for i, ae in enumerate(self.autoencoders):
            ae.load_state_dict(checkpoint['model_state_dicts'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dicts'][i])
            self.schedulers[i].load_state_dict(checkpoint['scheduler_state_dicts'][i])
            ae.to(self.device)
        self.scaler.load_state_dict(checkpoint['scaler'])