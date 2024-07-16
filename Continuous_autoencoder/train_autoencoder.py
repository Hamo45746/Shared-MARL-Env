import sys
import os
import numpy as np
import torch
from autoencoder import EnvironmentAutoencoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import Environment

# DATA_FOLDER = 'collected_data'
# COMBINED_DATA_FILE = 'combined_data.npy'

# def collect_data(env_config, config_path, num_samples):
#     env = Environment(config_path)
#     data = []
#     episode_data = env.run_simulation(max_steps=num_samples)
#     for step_data in episode_data:
#         for obs in step_data.values():
#             data.append(obs)
#     return np.array(data)

# def main():
#     config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    
#     # Collect data
#     num_samples = 1000
#     observation_data = collect_data(None, config_path, num_samples)
    
#     # Save collected data
#     os.makedirs(DATA_FOLDER, exist_ok=True)
#     np.save(os.path.join(DATA_FOLDER, COMBINED_DATA_FILE), observation_data)

#     # Train autoencoder
#     input_shape = observation_data.shape[1:]
#     autoencoder = EnvironmentAutoencoder(input_shape)
    
#     observation_data = torch.FloatTensor(observation_data)
#     autoencoder.train(observation_data, epochs=100, batch_size=32)
    
#     # Save the trained autoencoder model
#     save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_autoencoder.pth')
#     autoencoder.save(save_path)
    
#     print(f"Autoencoder training completed and model saved at {save_path}")

# if __name__ == "__main__":
#     main()



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
    
    # Assuming the shape of your observations is (5, 17, 17)
    input_shape = (5, 17, 17)
    
    train_autoencoder(data_path, model_save_path, input_shape)

if __name__ == "__main__":
    main()