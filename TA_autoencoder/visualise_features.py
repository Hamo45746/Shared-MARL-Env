import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import EnvironmentAutoencoder, LayerAutoencoder
from torch.autograd import Variable

def get_conv_layers(model):
    conv_layers = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
    return conv_layers

def visualize_features(autoencoder, layer_index, num_features=5, num_layers=4, input_shape=(1, 276, 155), num_iterations=30, learning_rate=0.1):
    device = next(autoencoder.parameters()).device
    conv_layers = get_conv_layers(autoencoder.autoencoders[layer_index].encoder)
    
    fig, axes = plt.subplots(num_layers, num_features, figsize=(15, 3*num_layers))
    fig.suptitle(f'Top {num_features} Features for Each Layer of Autoencoder {layer_index}')
    
    for layer, conv_layer in enumerate(conv_layers[:num_layers]):
        for i in range(num_features):
            # Start from random noise
            input_img = Variable(torch.randn(1, *input_shape).to(device), requires_grad=True)
            
            # Optimization loop
            for _ in range(num_iterations):
                optimizer = torch.optim.Adam([input_img], lr=learning_rate)
                optimizer.zero_grad()
                
                # Forward pass
                x = input_img
                for l in conv_layers[:layer+1]:
                    x = l(x)
                
                # Maximize activation of the i-th filter
                loss = -torch.mean(x[0, i])
                loss.backward()
                optimizer.step()
            
            # Normalize and convert to image
            feature = input_img.data.squeeze().cpu().numpy()
            feature = (feature - feature.min()) / (feature.max() - feature.min())
            
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
    
    # Move autoencoder to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)

    # Visualize features for each autoencoder
    for i in range(3):
        fig = visualize_features(autoencoder, i)
        fig.savefig(f'autoencoder_{i}_features.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()