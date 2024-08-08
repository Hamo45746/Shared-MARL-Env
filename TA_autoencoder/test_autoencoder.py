import os
import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import EnvironmentAutoencoder
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
H5_FOLDER = '/Volumes/T7 Shield/METR4911/TA_autoencoder_h5_data'
AUTOENCODER_FILE = 'trained_autoencoder.pth'

def find_suitable_h5_file(h5_folder):
    logging.info("Searching for suitable H5 file...")
    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5') and 'a14' in filename:
            logging.info(f"Found suitable file: {filename}")
            return os.path.join(h5_folder, filename)
    raise FileNotFoundError("No suitable H5 file found.")

def load_data_from_h5(h5_file, step=0, agent=0):
    logging.info(f"Loading data from {h5_file}")
    try:
        with h5py.File(h5_file, 'r') as f:
            step_data = f['data'][str(step)]
            agent_data = step_data[str(agent)]
            full_state = agent_data['full_state'][()]
        logging.info("Data loaded successfully")
        return full_state
    except Exception as e:
        logging.error(f"Error loading data from H5 file: {str(e)}")
        raise

def visualize_autoencoder_progress(autoencoder, input_data, layer, epoch, output_folder):
    logging.info(f"Starting visualization for layer {layer}, epoch {epoch}")
    start_time = time.time()
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(f'Layer {layer} - Epoch {epoch}')

    # Input
    logging.info("Processing input data")
    im_input = axes[0].imshow(input_data, cmap='viridis')
    axes[0].set_title('Input')
    plt.colorbar(im_input, ax=axes[0], fraction=0.046, pad=0.04)

    # Output
    logging.info("Encoding and decoding data")
    ae_index = min(layer, 2)  # 0 for layer 0, 1 for layers 1 and 2, 2 for layer 3
    with torch.no_grad():
        encoded = autoencoder.autoencoders[ae_index].encode(torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0))
        decoded = autoencoder.autoencoders[ae_index].decoder(encoded).cpu().numpy().squeeze()
    im_output = axes[1].imshow(decoded, cmap='viridis')
    axes[1].set_title('Output')
    plt.colorbar(im_output, ax=axes[1], fraction=0.046, pad=0.04)

    logging.info("Saving figure")
    plt.tight_layout()
    output_file = os.path.join(output_folder, f'layer_{layer}_epoch_{epoch}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    end_time = time.time()
    logging.info(f"Visualization for layer {layer}, epoch {epoch} completed in {end_time - start_time:.2f} seconds")
    logging.info(f"Saved visualization to {output_file}")

    return decoded

def save_layer_data_to_file(data, layer, epoch, output_folder, data_type):
    filename = os.path.join(output_folder, f'layer_{layer}_epoch_{epoch}_{data_type}.txt')
    np.savetxt(filename, data, fmt='%.4f')
    logging.info(f"Saved layer {layer} {data_type} data to {filename}")

def load_autoencoder_for_layer(path, input_shape, device, layer, epoch):
    logging.info(f"Loading autoencoder from {path}")
    try:
        autoencoder = EnvironmentAutoencoder(input_shape, device)
        checkpoint = torch.load(path, map_location=device)
        
        logging.info(f"Checkpoint keys: {checkpoint.keys()}")
        
        if epoch == 'final':
            for i, ae in enumerate(autoencoder.autoencoders):
                ae.load_state_dict(checkpoint['model_state_dicts'][i])
                ae.eval()
        else:
            ae_index = min(layer, 2)
            autoencoder.autoencoders[ae_index].load_state_dict(checkpoint['model_state_dicts'][0])
            autoencoder.autoencoders[ae_index].eval()
        
        logging.info("Autoencoder loaded successfully")
        return autoencoder
    except Exception as e:
        logging.error(f"Error loading autoencoder: {str(e)}")
        raise

def test_specific_autoencoder(autoencoder_path, h5_folder, output_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        # Find a suitable H5 file
        h5_file = find_suitable_h5_file(h5_folder)
        full_state = load_data_from_h5(h5_file, step=30)

        input_shape = full_state.shape
        logging.info(f"Full input shape: {input_shape}")

        # Load the specific autoencoder
        autoencoder = EnvironmentAutoencoder(input_shape, device)
        checkpoint = torch.load(autoencoder_path, map_location=device)
        
        if 'model_state_dicts' in checkpoint:
            # This is a full autoencoder save
            for i, ae in enumerate(autoencoder.autoencoders):
                ae.load_state_dict(checkpoint['model_state_dicts'][i])
                ae.to(device)
                ae.eval()
        else:
            # This is a single autoencoder save
            autoencoder.autoencoders[0].load_state_dict(checkpoint)
            autoencoder.autoencoders[0].to(device)
            autoencoder.autoencoders[0].eval()

        logging.info(f"Loaded autoencoder from {autoencoder_path}")

        os.makedirs(output_folder, exist_ok=True)

        # Test and visualize each layer
        for layer in range(4):
            ae_index = min(layer, 2)  # 0 for layer 0, 1 for layers 1 and 2, 2 for layer 3
            input_data = full_state[layer]

            logging.info(f"Layer {layer} - Input shape: {input_data.shape}")

            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f'Layer {layer} - Autoencoder Test')

            # Input
            im_input = axes[0].imshow(input_data, cmap='viridis')
            axes[0].set_title('Input')
            plt.colorbar(im_input, ax=axes[0], fraction=0.046, pad=0.04)

            # Output
            with torch.no_grad():
                input_tensor = torch.FloatTensor(input_data).unsqueeze(0).unsqueeze(0).to(device)
                logging.info(f"Layer {layer} - Input tensor shape: {input_tensor.shape}")
                
                # Use the forward method of the autoencoder
                decoded = autoencoder.autoencoders[ae_index](input_tensor)
                logging.info(f"Layer {layer} - Decoded shape: {decoded.shape}")
                
                decoded = decoded.squeeze().cpu().numpy()
                logging.info(f"Layer {layer} - Final decoded shape: {decoded.shape}")

            im_output = axes[1].imshow(decoded, cmap='viridis')
            axes[1].set_title('Reconstructed Output')
            plt.colorbar(im_output, ax=axes[1], fraction=0.046, pad=0.04)

            # Calculate and display MSE
            if input_data.shape != decoded.shape:
                logging.warning(f"Layer {layer} - Shape mismatch: input {input_data.shape}, decoded {decoded.shape}")
                mse = np.nan
            else:
                mse = np.mean((input_data - decoded) ** 2)
            
            plt.suptitle(f'Layer {layer} - Autoencoder Test (MSE: {mse:.6f})')

            # Save the figure
            plt.tight_layout()
            output_file = os.path.join(output_folder, f'test_layer_{layer}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logging.info(f"Saved visualization for layer {layer} to {output_file}")
            logging.info(f"MSE for layer {layer}: {mse:.6f}")

        print(f"Test complete. Visualizations saved in {output_folder}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.exception("Exception details:")
        raise

# You can keep your existing main() function and add this new one
def main_test_specific():
    H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
    AUTOENCODER_FILE = 'autoencoder_0_best.pth'  # Update this to the file you want to test
    OUTPUT_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data/training_visualisations'  # Update this path

    autoencoder_path = os.path.join(H5_FOLDER, AUTOENCODER_FILE)
    test_specific_autoencoder(autoencoder_path, H5_FOLDER, OUTPUT_FOLDER)

def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        h5_file = find_suitable_h5_file(H5_FOLDER)
        full_state = load_data_from_h5(h5_file, step=30)

        input_shape = full_state.shape
        logging.info(f"Input shape: {input_shape}")

        output_folder = os.path.join(H5_FOLDER, 'autoencoder_visualizations')
        os.makedirs(output_folder, exist_ok=True)
        logging.info(f"Output folder: {output_folder}")

        epochs_to_visualize = [10, 50, 'final']

        for epoch in epochs_to_visualize:
            for layer in range(4):
                if epoch == 'final':
                    autoencoder_path = os.path.join(H5_FOLDER, AUTOENCODER_FILE)
                else:
                    ae_index = min(layer, 2)
                    autoencoder_path = os.path.join(H5_FOLDER, f"autoencoder_{ae_index}_epoch_{epoch}.pth")
                
                if not os.path.exists(autoencoder_path):
                    logging.warning(f"Autoencoder file not found: {autoencoder_path}")
                    continue

                load_start = time.time()
                autoencoder = load_autoencoder_for_layer(autoencoder_path, input_shape, device, layer, epoch)
                load_end = time.time()
                logging.info(f"Loading autoencoder took {load_end - load_start:.2f} seconds")

                vis_start = time.time()
                output_data = visualize_autoencoder_progress(autoencoder, full_state[layer], layer, epoch, output_folder)
                vis_end = time.time()
                logging.info(f"Visualization took {vis_end - vis_start:.2f} seconds")

                # Save layer data to separate files - to check input
                # if layer == 2:
                    # save_layer_data_to_file(full_state[layer], layer, epoch, output_folder, "input")
                    # save_layer_data_to_file(output_data, layer, epoch, output_folder, "output")

        end_time = time.time()
        logging.info(f"Total execution time: {end_time - start_time:.2f} seconds")
        print(f"Visualizations and data files saved in {output_folder}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # main()
    main_test_specific()