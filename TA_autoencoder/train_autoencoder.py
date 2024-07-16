import os
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from tqdm import tqdm
from autoencoder import EnvironmentAutoencoder
import multiprocessing

# Constants
H5_FOLDER = '/Volumes/T7 Shield/METR4911/TA_autoencoder_h5_data'
H5_PROGRESS_FILE = 'h5_collection_progress.txt'
AUTOENCODER_FILE = 'trained_autoencoder.pth'
TRAINING_STATE_FILE = 'training_state.pth'

class H5Dataset(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files
        self.cumulative_sizes = self._get_cumulative_sizes()

    def _get_cumulative_sizes(self):
        sizes = []
        total = 0
        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                total += len(f['data'])
            sizes.append(total)
        return sizes

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if file_idx == 0:
            internal_idx = idx
        else:
            internal_idx = idx - self.cumulative_sizes[file_idx - 1]
        with h5py.File(self.h5_files[file_idx], 'r') as f:
            data = f['data'][internal_idx]
        return torch.FloatTensor(data)

def load_progress():
    progress_file = os.path.join(H5_FOLDER, H5_PROGRESS_FILE)
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(f.read().splitlines())
    return set()

def main():
    # Set up CPU optimizations
    torch.set_num_threads(multiprocessing.cpu_count())
    torch.set_num_interop_threads(multiprocessing.cpu_count())

    # Load progress and prepare dataset
    completed_configs = load_progress()
    h5_files = [os.path.join(H5_FOLDER, filename) for filename in completed_configs]
    dataset = H5Dataset(h5_files)

    # Get input shape from the first item in the dataset
    with h5py.File(h5_files[0], 'r') as f:
        input_shape = f['data'][0].shape
    print(f"Input shape: {input_shape}")

    # Initialize autoencoder
    autoencoder = EnvironmentAutoencoder(input_shape)

    # Create DataLoader
    batch_size = 32  # Reduced batch size for CPU
    num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU core free
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=False)

    # Training parameters
    epochs = 100

    # Train the autoencoder
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        avg_loss = autoencoder.train(dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(H5_FOLDER, f"autoencoder_checkpoint_epoch_{epoch+1}.pth")
            autoencoder.save(checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Save the final model
    autoencoder.save(os.path.join(H5_FOLDER, AUTOENCODER_FILE))
    print(f"Autoencoder training completed and model saved at {os.path.join(H5_FOLDER, AUTOENCODER_FILE)}")

if __name__ == "__main__":
    main()