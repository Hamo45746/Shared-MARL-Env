import sys
import os
import numpy as np
import torch
from autoencoder import EnvironmentAutoencoder

def train_autoencoder(data_path, model_save_path, layer_name):
    # Load the combined dataset
    observation_data = np.load(data_path)
    print(f"Loaded {layer_name} data with shape: {observation_data.shape}")

    # Ensure the data is in the right format for PyTorch (float32)
    observation_data = torch.FloatTensor(observation_data).unsqueeze(1)
    print(f"Converted {layer_name} data to torch tensor with shape: {observation_data.shape}")

    # Create the autoencoder model
    autoencoder = EnvironmentAutoencoder((17, 17), encoded_dim=32)

    # Train the autoencoder
    autoencoder.add_layer(layer_name, (17, 17), encoded_dim=32)
    autoencoder.train({layer_name: observation_data}, layer_name, epochs=300, batch_size=32)
    
    # Save the trained model
    autoencoder.save(model_save_path, layer_name)


def main():
    for layer_name in ["map_view", "agent", "target", "jammer"]:
        data_path = f'outputs/combined_data_{layer_name}.npy'
        model_save_path = f'outputs/trained_autoencoder_{layer_name}.pth'
        if os.path.exists(data_path):
            train_autoencoder(data_path, model_save_path, layer_name)
        else:
            print(f"Data file for {layer_name} does not exist at {data_path}")

if __name__ == "__main__":
    main()