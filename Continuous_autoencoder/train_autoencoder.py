import sys
import os
import numpy as np
import torch
from autoencoder import EnvironmentAutoencoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rllib_env import Environment

def train_autoencoder(data_path, model_save_path, input_shape):
    # Load the combined dataset
    observation_data = np.load(data_path)
    
    # Ensure the data is in the right format for PyTorch (float32)
    observation_data = torch.FloatTensor(observation_data)

    # Create the autoencoder model
    autoencoder = EnvironmentAutoencoder(input_shape)
    
    # Train the autoencoder
    autoencoder.train(observation_data, epochs=150, batch_size=32)
    
    # Save the trained model
    autoencoder.save(model_save_path)

def main():
    data_path = 'outputs/combined_data.npy'
    model_save_path = 'outputs/trained_autoencoder.pth'
    
    input_shape = (5, 17, 17)
    
    train_autoencoder(data_path, model_save_path, input_shape)

if __name__ == "__main__":
    main()