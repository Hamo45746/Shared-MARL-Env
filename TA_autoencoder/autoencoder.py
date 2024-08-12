import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.nn.functional as F
import numpy as np
import math

def ceildiv(a, b):
    return -(a // -b)

def calculate_elu_gain(negative_slope=1.0):
    return math.sqrt(1.55)  # Theoretical gain for ELU

def kaiming_elu_init_(tensor, a=1.0, mode='fan_in', nonlinearity='elu'):
    fan = nn.init._calculate_correct_fan(tensor, mode)
    gain = calculate_elu_gain(a)
    std = gain / math.sqrt(fan)
    with torch.no_grad():
        return tensor.normal_(0, std)

class LayerAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(LayerAutoencoder, self).__init__()
        self.input_shape = input_shape  # (276, 155)

        # Encoder convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        # Calculate the output size of the last convolutional layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            conv_output = self.encoder_conv(dummy_input)
            self.flatten_size = conv_output.numel()
            self.conv_output_shape = conv_output.shape[1:]

        # Encoder linear layers
        self.encoder_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.flatten_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, self.conv_output_shape),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.final_activation = nn.Hardtanh(min_val=-20, max_val=0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder_conv(x)
        encoded = self.encoder_linear(x)
        decoded = self.decoder(encoded)
        
        # Interpolate to match input size
        decoded = F.interpolate(decoded, size=self.input_shape, mode='bilinear', align_corners=False)
        
        # Apply thresholding to mitigate interpolation effects
        background_mask = (decoded <= -19.9).float()
        decoded = decoded * (1 - background_mask) + (-20) * background_mask
        return self.final_activation(decoded)

    def encode(self, x):
        x = self.encoder_conv(x)
        return self.encoder_linear(x)

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

    def custom_loss(self, recon_x, x, layer):
        if layer == 0:  # Binary case (0/1)
            # return F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean')
            return F.mse_loss(recon_x, x, reduction='mean')
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
            background_weight = 0.01  # Small weight for background
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
        ae = self.autoencoders[layer].to(self.device)
        optimizer = self.optimizers[layer]
        ae.train()

        layer_input = batch[f'layer_{layer}'].to(self.device)
        
        with amp.autocast():
            outputs = ae(layer_input.unsqueeze(1))  # Add channel dimension
            loss = self.custom_loss(outputs.squeeze(1), layer_input, layer)  # Removed channel dimension for loss calculation

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