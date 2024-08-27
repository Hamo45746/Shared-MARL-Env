import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

class ScaledSigmoid(nn.Module):
    def __init__(self, min_val, max_val):
        super(ScaledSigmoid, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(x)

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, input_shape, encoded_dim=32, activation_fn=nn.Tanh()):
        super(ConvAutoencoder, self).__init__()
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.encoded_dim = encoded_dim

         # Calculate intermediate sizes
        flattened_size = 256 * 2 * 2
        intermediate_size1 = flattened_size // 4
        intermediate_size2 = intermediate_size1 // 4

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(flattened_size, intermediate_size1),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_size1, intermediate_size2),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_size2, encoded_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, intermediate_size2),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_size2, intermediate_size1),
            nn.LeakyReLU(0.2),
            nn.Linear(intermediate_size1, flattened_size),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, (256, 2, 2)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=0)
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def decode(self, x):
        decoded = self.decoder(x)
        return decoded

class EnvironmentAutoencoder:
    def __init__(self, input_shape, encoded_dim=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoders = {}
        self.optimizers = {}
        self.schedulers = {}
        #self.criterion = nn.MSELoss()
        self.criterion_custom = CustomLoss(zero_weight=10.0)
        self.criterion_standard = nn.MSELoss()
        
        # Initialize dictionaries to store original min and max values
        self.original_min = {}
        self.original_max = {}

    def add_layer(self, layer_name, input_shape, encoded_dim=32):

        if layer_name == "map_view":
            activation_fn = nn.Sigmoid()
        else:
            activation_fn = ScaledSigmoid(-20, 0)

        self.autoencoders[layer_name] = ConvAutoencoder(1, input_shape, encoded_dim, activation_fn).to(self.device)
        self.optimizers[layer_name] = optim.Adam(self.autoencoders[layer_name].parameters(), lr=0.0005)
        self.schedulers[layer_name] = optim.lr_scheduler.ReduceLROnPlateau(self.optimizers[layer_name], 'min', patience=9, factor=0.5, min_lr=1e-5)

    def train(self, data, layer_name, epochs=10, batch_size=32, validation_split=0.15):
        # Split data into training and validation sets
        train_data, val_data = train_test_split(data[layer_name], test_size=validation_split, random_state=42)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
        
        print(f"Training on data for layer: {layer_name} with {len(train_loader)} batches. Validation set has {len(val_loader)} batches.")
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.autoencoders[layer_name].train()
            total_train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizers[layer_name].zero_grad()
                encoded, decoded = self.autoencoders[layer_name](batch)
                if layer_name == "map_view":
                    loss = self.criterion_standard(decoded, batch)
                else:
                    loss = self.criterion_custom(decoded, batch)
                loss.backward()
                self.optimizers[layer_name].step()
                total_train_loss += loss.item()
            
            average_train_loss = total_train_loss / len(train_loader)
            train_losses.append(average_train_loss)

            # Validation phase
            self.autoencoders[layer_name].eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(self.device)
                    _, decoded = self.autoencoders[layer_name](val_batch)
                    if layer_name == "map_view":
                        val_loss = self.criterion_standard(decoded, val_batch)
                    else:
                        val_loss = self.criterion_custom(decoded, val_batch)
                    total_val_loss += val_loss.item()

            average_val_loss = total_val_loss / len(val_loader)
            val_losses.append(average_val_loss)
            
            self.schedulers[layer_name].step(average_val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {average_train_loss:.6f}, Validation Loss: {average_val_loss:.6f}")
     
            if (epoch + 1) % 60 == 0:
                self.visualize_reconstructions(val_loader, layer_name, epoch)

            if (epoch + 1) % 400 == 0:
                self.visualize_reconstructions(val_loader, layer_name, epoch)

        # Plot the training and validation losses
        self.plot_loss(train_losses, val_losses, layer_name)

    def plot_loss(self, train_losses, val_losses, layer_name):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Training and Validation Loss for {layer_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join('outputs/plots', f'lossplot_{layer_name}.png')
        plt.savefig(plot_path)

    def encode_state(self, state):
        encoded_state = {}
        for layer_name, layer_data in state.items():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(layer_data).unsqueeze(0).unsqueeze(0).to(self.device)
                encoded = self.autoencoders[layer_name].encode(state_tensor).cpu().numpy().squeeze()
                encoded_state[layer_name] = {'encoded': encoded}
        return encoded_state

    def save(self, path, layer_name):
        torch.save(self.autoencoders[layer_name].state_dict(), path)

    def load(self, path, layer_name):
        self.autoencoders[layer_name].load_state_dict(torch.load(path))
        self.autoencoders[layer_name].eval()

    def decode_state(self, state):
        decoded_state = {}
        for layer_name, encoded_data in state.items():
            encoded_tensor = torch.FloatTensor(encoded_data['encoded']).unsqueeze(0).to(self.device)
            decoded = self.autoencoders[layer_name].decode(encoded_tensor).detach().cpu().numpy().squeeze()
            decoded_state[layer_name] = decoded

        return decoded_state
    

    def visualize_reconstructions(self, dataloader, layer_name, epoch):
        self.autoencoders[layer_name].eval()
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                encoded, decoded = self.autoencoders[layer_name](batch)
                batch = batch.cpu().numpy()
                decoded = decoded.cpu().numpy()

                # Visualize a few samples
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                for i in range(5):
                    if layer_name == 'map_view':
                        axes[0, i].imshow(batch[i].squeeze(), cmap='grey', vmin=0, vmax=1)
                    else: 
                        axes[0, i].imshow(batch[i].squeeze(), cmap='grey', vmin=-20, vmax=0)
                    axes[0, i].set_title('Original')
                    axes[0, i].axis('off')
                    if layer_name == 'map_view':
                        axes[1, i].imshow(decoded[i].squeeze(), cmap='grey', vmin=0, vmax=1)
                    else:
                        axes[1, i].imshow(decoded[i].squeeze(), cmap='grey', vmin=-20, vmax=0)
                    axes[1, i].set_title('Reconstructed')
                    axes[1, i].axis('off')

                plt.suptitle(f'Reconstructions at Epoch {epoch} for {layer_name}')
                plot_path = os.path.join('outputs/plots', f'reconstructions_{layer_name}_epoch_{epoch}.png')
                plt.savefig(plot_path)
                plt.close(fig)
                break
        self.autoencoders[layer_name].train()

# Custom Loss Function
class CustomLoss(nn.Module):
    def __init__(self, zero_weight=10.0):
        super(CustomLoss, self).__init__()
        self.zero_weight = zero_weight

    def forward(self, predictions, targets):
        # Compute the standard MSE loss
        mse_loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Create a mask where the target is zero
        zero_mask = (targets == 0).float()
        
        # Apply the higher weight to the zero_mask areas
        weighted_loss = mse_loss * (1 + self.zero_weight * zero_mask)
        
        # Return the mean of the weighted loss
        return weighted_loss.mean()