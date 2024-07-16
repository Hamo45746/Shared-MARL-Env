import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(ConvAutoencoder, self).__init__()
        self.input_shape = input_shape

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Calculate the size of the encoder output
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            encoded_size = self.encoder(x).shape[1:]
        self.encoded_size = encoded_size

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

class EnvironmentAutoencoder:
    def __init__(self, input_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder = ConvAutoencoder(input_shape).to(self.device)
        self.optimizer = optim.Adam(self.autoencoder.parameters())
        self.criterion = nn.MSELoss()

    def train(self, data, start_epoch=0, epochs=1, batch_size=32):
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

        total_loss = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.autoencoder(batch)
            
            # Ensure the output has the same size as the input
            outputs = nn.functional.interpolate(outputs, size=batch.shape[2:], mode='bilinear', align_corners=False)
            
            loss = self.criterion(outputs, batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def encode_state(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            encoded = self.autoencoder.encode(state_tensor)
        return encoded.cpu().numpy().squeeze()

    def save(self, path):
        torch.save(self.autoencoder.state_dict(), path)

    def load(self, path):
        self.autoencoder.load_state_dict(torch.load(path))
        self.autoencoder.eval()