import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import EnvironmentAutoencoder, LayerAutoencoder

def visualise_features(autoencoder, layer_index, num_features=5):
    # Get the first convolutional layer of the encoder
    conv_layer = autoencoder.autoencoders[layer_index].encoder[0]
    
    # Get the weights of the convolutional layer
    weights = conv_layer.weight.data.cpu().numpy()
    
    # Normalise the weights
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    
    # Create a figure to display the features
    fig, axes = plt.subplots(1, num_features, figsize=(15, 3))
    fig.suptitle(f'Top {num_features} Features for Autoencoder {layer_index}')
    
    # For each of the top features
    for i in range(num_features):
        # Get the i-th filter
        feature = weights[i, 0]
        
        # Display the feature
        im = axes[i].imshow(feature, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i+1}')
    
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    return fig

def main():
    # Load your autoencoders
    autoencoder = EnvironmentAutoencoder()
    autoencoder.load_all_autoencoders('/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data/AE_save_06_09')

    # Visualise features for each autoencoder
    for i in range(3):
        fig = visualise_features(autoencoder, i)
        fig.savefig(f'autoencoder_{i}_features.png')
        plt.close(fig)

if __name__ == "__main__":
    main()