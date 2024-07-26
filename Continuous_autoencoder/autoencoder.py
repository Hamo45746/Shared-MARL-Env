import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp

class ConvAutoencoder(nn.Module):
    def __init__(self, input_channels, input_shape, encoded_dim=32):
        super(ConvAutoencoder, self).__init__()
        self.input_channels = input_channels
        self.input_shape = input_shape
        self.encoded_dim = encoded_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the size of the encoder output
        with torch.no_grad():
            x = torch.zeros(1, input_channels, *input_shape)
            flattened_size = self.encoder(x).shape[1]

        print(f"Flattened size: {flattened_size}")

        # Linear layers for encoding and decoding
        self.encoder_linear = nn.Linear(flattened_size, encoded_dim)
        self.decoder_linear = nn.Linear(encoded_dim, flattened_size)
        unflatten_shape = (128, 3, 3)
        # Decoder
        self.decoder = nn.Sequential(
            #nn.Linear(encoded_dim, flattened_size),
            nn.Unflatten(1, unflatten_shape),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

    def encode(self, x):
        encoded = self.encoder(x)
        encoded = self.encoder_linear(encoded)
        return encoded
    
    def decode(self, x):
        decoded = self.decoder_linear(x)
        decoded = self.decoder(decoded)
        return decoded

class EnvironmentAutoencoder:
    def __init__(self, input_shape, layer_name, encoded_dim=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoders = {
            layer_name: ConvAutoencoder(1, input_shape, encoded_dim).to(self.device)
        }
        self.optimizers = {layer_name: optim.Adam(self.autoencoders[layer_name].parameters())}
        self.criterion = nn.MSELoss()

    def train(self, data, layer_name, epochs=4, batch_size=32):
        dataloader = torch.utils.data.DataLoader(data[layer_name], batch_size=batch_size, shuffle=True)
        print(f"Training on data for layers: {layer_name}")
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(self.device)
                self.optimizers[layer_name].zero_grad()
                encoded, decoded = self.autoencoders[layer_name](batch)
                loss = self.criterion(decoded, batch)
                loss.backward()
                self.optimizers[layer_name].step()
                total_loss += loss.item()

            if (epoch + 1) % 2 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss}")

    def encode_state(self, state):
        encoded_state = {}
        for key, layer in state.items():
            with torch.no_grad():
                state_tensor = torch.FloatTensor(layer).unsqueeze(0).unsqueeze(0).to(self.device)
                encoded_state[key] = self.autoencoders[key].encode(state_tensor).cpu().numpy().squeeze()
        return encoded_state
    
    def decode_state(self, state):
        decoded_state = {}
        for key, encoded in state.items():
            with torch.no_grad():
                encoded_tensor = torch.FloatTensor(encoded).unsqueeze(0).to(self.device)
                decoded_state[key] = self.autoencoders[key].decode(encoded_tensor).cpu().numpy().squeeze()
        return decoded_state 

    def save(self, path, layer_name):
        torch.save(self.autoencoders[layer_name].state_dict(), path)

    def load(self, path, layer_name):
        self.autoencoders[layer_name].load_state_dict(torch.load(path))
        self.autoencoders[layer_name].eval()