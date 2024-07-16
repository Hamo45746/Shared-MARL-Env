import torch
import torch.nn as nn
import torch.optim as optim

class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape):
        super(ConvAutoencoder, self).__init__()
        self.input_shape = input_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1)
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
        print(f"Using device: {self.device}")
        self.autoencoder = ConvAutoencoder(input_shape).to(self.device)
        self.optimizer = optim.Adam(self.autoencoder.parameters())
        self.criterion = nn.MSELoss()

    def train(self, dataloader):
        self.autoencoder.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.autoencoder(batch)
            
            # Ensure output size matches input size
            if outputs.size() != batch.size():
                outputs = nn.functional.interpolate(outputs, size=batch.size()[2:], mode='bilinear', align_corners=False)
            
            loss = self.criterion(outputs, batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def encode_state(self, state):
        self.autoencoder.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            encoded = self.autoencoder.encode(state_tensor)
        return encoded.cpu().numpy().squeeze()

    def save(self, path):
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_shape': self.autoencoder.input_shape
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.autoencoder.eval()