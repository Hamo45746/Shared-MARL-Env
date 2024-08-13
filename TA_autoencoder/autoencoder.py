import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


class LayerAutoencoder(nn.Module):
    def __init__(self, is_map=False):
        super(LayerAutoencoder, self).__init__()
        self.is_map = is_map

        # Encoder convolutional layers (no padding)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
        )

        # Calculate the flattened size
        self.flattened_size = 256 * 16 * 8  # 32,768

        # Encoder linear layers with more gradual reduction
        self.encoder_linear = nn.Sequential(
            nn.Linear(self.flattened_size, 8192),
            nn.LeakyReLU(0.2),
            nn.Linear(8192, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 128)
        )

        # Decoder linear layers with gradual expansion
        self.decoder_linear = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 8192),
            nn.LeakyReLU(0.2),
            nn.Linear(8192, self.flattened_size),
            nn.LeakyReLU(0.2)
        )

        # Decoder convolutional layers
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Encoder
        x = nn.Flatten(self.encoder_conv(x))
        encoded = self.encoder_linear(x)

        # Decoder
        x = nn.Unflatten(self.decoder_linear(encoded))
        x = self.decoder_conv(x)
        
        # Ensure output size is 276x155 using interpolation
        x = F.interpolate(x, size=(276, 155), mode='bilinear', align_corners=False)

        if not self.is_map:
            x = torch.clamp(x, min=-20, max=0)
            
        return x

    def encode(self, x):
        x = self.encoder_conv(x)
        x = nn.Flatten(x)
        return self.encoder_linear(x)

    def decode(self, x):
        x = nn.Unflatten(self.decoder_linear(x))
        x = self.decoder_conv(x)
        # Ensure output size is 276x155 using interpolation
        x = F.interpolate(x, size=(276, 155), mode='bilinear', align_corners=False)
        if not self.is_map:
            x = torch.clamp(x, min=-20, max=0)
        return x

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
        self.schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, verbose=True) for opt in self.optimizers]
        self.scaler = GradScaler()

    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Map layer (Binary Cross Entropy loss with logits)
            return F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
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
        
        self.schedulers[layer].step(loss.item())
        
        return loss.item()

    def encode_state(self, state):
        with torch.no_grad():
            encoded_layers = []
            for i, ae in enumerate(self.autoencoders):
                ae.to(self.device)
                ae.eval()
                layer_input = torch.FloatTensor(state[i]).unsqueeze(0).unsqueeze(0).to(self.device)
                encoded_layer = ae.encode(layer_input)
                encoded_layers.append(encoded_layer.cpu().numpy().squeeze())
                ae.cpu()
            torch.cuda.empty_cache()
        return encoded_layers

    def decode_state(self, encoded_state):
        with torch.no_grad():
            decoded_layers = []
            for i, ae in enumerate(self.autoencoders):
                ae.to(self.device)
                ae.eval()
                encoded_layer = torch.FloatTensor(encoded_state[i]).unsqueeze(0).to(self.device)
                decoded_layer = ae(encoded_layer.unsqueeze(2).unsqueeze(3))
                decoded_layers.append(decoded_layer.cpu().numpy().squeeze())
                ae.cpu()
            torch.cuda.empty_cache()
        return decoded_layers

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