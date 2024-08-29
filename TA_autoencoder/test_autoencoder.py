import os
import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import EnvironmentAutoencoder, LayerAutoencoder
import time
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
H5_FOLDER = '/Volumes/T7 Shield/METR4911/TA_autoencoder_h5_data'
AUTOENCODER_FILE = 'trained_autoencoder.pth'

def find_suitable_h5_file(h5_folder):
    logging.info("Searching for suitable H5 file...")
    for filename in os.listdir(h5_folder):
        if filename.endswith('.h5') and 't90' in filename and 'image_2' in filename:
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

def visualise_data(filepath, savepath, step=0, agent=0):
    full_state = load_data_from_h5(filepath)

    fig, axes = plt.subplots(1, 4, figsize=(20, 10))
    fig.suptitle(f'Data visualisation. File: {filepath}, step: {step}, agent: {agent}')

    for layer in range(0, 4):
        input_data = full_state[layer]
        im_input = axes[0].imshow(input_data, cmap='viridis')
        if layer == 0:
            axes[layer].set_title('Map')
        elif layer == 1:
            axes[layer].set_title('Agents')
        elif layer == 2:
            axes[layer].set_title('Targets')
        else:
            axes[layer].set_title('Jammers')
        plt.colorbar(im_input, ax=axes[layer], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.close(fig)


def test_specific_autoencoder(autoencoder, h5_folder, output_folder, autoencoder_index=None, epoch=None):
    device = autoencoder.device
    logging.info(f"Using device: {device}")

    try:
        # Find a suitable H5 file
        # h5_file = find_suitable_h5_file(h5_folder)
        # filename = 'data_mcity_image_1_s5_t90_j89_a10.h5'
        # filename = 'data_mcity_image_1_s2485_t90_j0_a1.h5'
        filename = 'test_data/data_mrand_s1176_t55_j0_a8.h5'
        h5_file = os.path.join(h5_folder, filename)
        full_state = load_data_from_h5(h5_file, step=0, agent=0)

        input_shape = full_state.shape
        logging.info(f"Full input shape: {input_shape}")

        os.makedirs(output_folder, exist_ok=True)

        # Determine which layers to visualize
        if autoencoder_index is None:
            layers_to_visualize = range(4)
        elif autoencoder_index == 0:
            layers_to_visualize = [0]
        elif autoencoder_index == 1:
            layers_to_visualize = [1, 2]
        else:
            layers_to_visualize = [3]

        # Test and visualize each layer
        for layer in layers_to_visualize:
            if layer >= 2:
                ae_index = layer - 1
            else:
                ae_index = layer
            input_data = full_state[layer]

            logging.info(f"Layer {layer} - Input shape: {input_data.shape}")

            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f'Layer {layer} - Autoencoder Test')

            # Input
            if layer == 0:
                im_input = axes[0].imshow(input_data, cmap='viridis', vmin=0, vmax=1)
            else:
                im_input = axes[0].imshow(input_data, cmap='viridis', vmin=-20, vmax=0)
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
            if layer == 0:
                im_output = axes[1].imshow(decoded, cmap='viridis', vmin=-0, vmax=1)
            else:
                im_output = axes[1].imshow(decoded, cmap='viridis', vmin=-20, vmax=0)
            axes[1].set_title('Reconstructed Output')
            plt.colorbar(im_output, ax=axes[1], fraction=0.046, pad=0.04)

            # Calculate and display MSE
            if input_data.shape != decoded.shape:
                logging.warning(f"Layer {layer} - Shape mismatch: input {input_data.shape}, decoded {decoded.shape}")
                mse = np.nan
            else:
                mse = np.mean((input_data - decoded) ** 2)
            
            epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
            plt.suptitle(f'Layer {layer} - {epoch_str}Autoencoder Test (MSE: {mse:.6f})')

            # Save the figure
            plt.tight_layout()
            epoch_prefix = f"epoch_{epoch}_" if epoch is not None else ""
            output_file = os.path.join(output_folder, f'{epoch_prefix}test_layer_{layer}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            logging.info(f"Saved visualization for layer {layer} to {output_file}")
            logging.info(f"MSE for layer {layer}: {mse:.6f}")

        print(f"Test complete. Visualizations saved in {output_folder}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.exception("Exception details:")
        raise

def main_test_data():
    H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
    H5_FILE = 'data_mcity_image_2_s1_t90_j0_a15.h5'
    OUTPUT_FILE = 'data_test_vis.png'
    savepath = os.path.join(H5_FOLDER, OUTPUT_FILE)
    path = os.path.join(H5_FOLDER, H5_FILE)
    visualise_data(path, savepath, step = 50)


def main_test_specific():
    H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
    # AUTOENCODER_FILE = 'AE_save_23_08/autoencoder_0_best.pth'  # Update this to the Autoencoder to test
    AUTOENCODER_FILE = 'autoencoder_0_best.pth'
    OUTPUT_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data/training_visualisations'  # Update this path

    autoencoder_path = os.path.join(H5_FOLDER, AUTOENCODER_FILE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = EnvironmentAutoencoder(device)
    autoencoder.load(autoencoder_path)
    test_specific_autoencoder(autoencoder, H5_FOLDER, OUTPUT_FOLDER, autoencoder_index=0)
    
    
def main_test_autoencoder_performance():
    # Constants
    H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
    TEST_SET_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data/test_data'
    AE_SAVE_FOLDER = 'AE_save_23_08'
    TEST_SET_SIZE = 10  # Number of H5 files to use for testing
    TRAIN_SET_SIZE = 20  # Number of H5 files to use for training comparison
    SAMPLES_PER_FILE = 100  # Number of samples to take from each file

    def calculate_mse(original, reconstructed):
        return np.mean((original - reconstructed) ** 2)

    def load_autoencoders(device):
        autoencoders = []
        for i in range(3):  # Load the three separate autoencoders
            ae_path = os.path.join(H5_FOLDER, AE_SAVE_FOLDER, f"autoencoder_{i}_best.pth")
            checkpoint = torch.load(ae_path, map_location=device)
            ae = LayerAutoencoder(is_map=(i == 0)).to(device)
            ae.load_state_dict(checkpoint['model_state_dicts'][i])
            ae.eval()
            autoencoders.append(ae)
        return autoencoders

    def process_file(file_path, autoencoders, device):
        with h5py.File(file_path, 'r') as f:
            data = f['data']
            steps = list(data.keys())
            selected_steps = np.random.choice(steps, min(len(steps), SAMPLES_PER_FILE), replace=False)
            
            mse_losses = {0: [], 1: [], 2: [], 3: []}
            
            for step in selected_steps:
                step_data = data[step]
                agents = list(step_data.keys())
                
                for agent in agents:
                    full_state = step_data[agent]['full_state'][()]
                
                    for layer in range(4):
                        if layer >= 2:
                            ae_index = layer - 1
                        else:
                            ae_index = layer
                        # ae_index = min(layer, 2)
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(full_state[layer]).unsqueeze(0).unsqueeze(0).to(device)
                            reconstructed = autoencoders[ae_index](input_tensor).cpu().numpy().squeeze()
                        
                        mse = calculate_mse(full_state[layer], reconstructed)
                        mse_losses[layer].append(mse)
            
            return {layer: np.mean(losses) for layer, losses in mse_losses.items()}

    # Load the trained autoencoders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoders = load_autoencoders(device)

    # Get all H5 files
    all_h5_files = [f for f in os.listdir(H5_FOLDER) if f.endswith('.h5')]
    test_h5_files = [f for f in os.listdir(TEST_SET_FOLDER) if f.endswith('.h5')]

    # Randomly select files for test and train sets
    np.random.shuffle(all_h5_files)
    test_files = test_h5_files[:TEST_SET_SIZE]
    train_files = all_h5_files[:TRAIN_SET_SIZE]

    # Process test set
    print("Processing test set...")
    test_results = {0: [], 1: [], 2: [], 3: []}
    for file in tqdm(test_files):
        file_path = os.path.join(TEST_SET_FOLDER, file)
        file_results = process_file(file_path, autoencoders, device)
        for layer, mse in file_results.items():
            test_results[layer].append(mse)

    # Process train set
    print("Processing train set...")
    train_results = {0: [], 1: [], 2: [], 3: []}
    for file in tqdm(train_files):
        file_path = os.path.join(H5_FOLDER, file)
        file_results = process_file(file_path, autoencoders, device)
        for layer, mse in file_results.items():
            train_results[layer].append(mse)

    # Calculate and print average MSE for each layer
    print("\nAverage MSE Results:")
    # logging.info("\nAverage MSE Results:")
    print("Layer | Test Set | Train Set")
    # logging.info("Layer | Test Set | Train Set")
    print("------|----------|----------")
    # logging.info("------|----------|----------")
    for layer in range(4):
        test_mse = np.mean(test_results[layer])
        train_mse = np.mean(train_results[layer])
        print(f"{layer}     | {test_mse:.6f} | {train_mse:.6f}")
        # logging.info(f"{layer}     | {test_mse:.6f} | {train_mse:.6f}")
    

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
                visualize_autoencoder_progress(autoencoder, full_state[layer], layer, epoch, output_folder)
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
    # main_test_data()
    # main_test_autoencoder_performance()