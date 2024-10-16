import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import EnvironmentAutoencoder, LayerAutoencoder

def get_conv_layers(model):
    conv_layers = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
    return conv_layers

def visualize_features(autoencoder, layer_index, num_features=5, num_layers=4):
    conv_layers = get_conv_layers(autoencoder.autoencoders[layer_index].encoder)
    
    fig, axes = plt.subplots(num_layers, num_features, figsize=(15, 3*num_layers))
    fig.suptitle(f'Top {num_features} Features for Each Layer of Autoencoder {layer_index}')
    
    for layer, conv_layer in enumerate(conv_layers[:num_layers]):
        weights = conv_layer.weight.data.cpu().numpy()
        
        # For multi-channel inputs, take the average across input channels
        if weights.shape[1] > 1:
            weights = np.mean(weights, axis=1)
        
        # Normalize the weights
        weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        for i in range(num_features):
            feature = weights[i]
            im = axes[layer, i].imshow(feature, cmap='viridis')
            axes[layer, i].axis('off')
            axes[layer, i].set_title(f'Layer {layer+1}, Feature {i+1}')
    
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    plt.tight_layout()
    return fig

def main():
    # Load your autoencoders
    autoencoder = EnvironmentAutoencoder()
    autoencoder.load_all_autoencoders('/path/to/your/autoencoder/folder')

    # Visualize features for each autoencoder
    for i in range(3):
        fig = visualize_features(autoencoder, i)
        fig.savefig(f'autoencoder_{i}_features.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()