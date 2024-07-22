import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
import multiprocessing as mp
from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from autoencoder import EnvironmentAutoencoder

# Constants
H5_FOLDER = '/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data'
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
            full_state = data['full_state']
            local_obs = data['local_obs']
        return {
            'full_state': torch.FloatTensor(full_state),
            'local_obs': torch.FloatTensor(local_obs)
        }

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def collect_data_for_config(config, config_path, steps_per_episode, h5_folder):
    env = Environment(config_path)
    env.config.update(config)
    env.num_agents = config['n_agents']
    env.initialise_agents()
    env.initialise_targets()
    env.initialise_jammers()
    env.update_global_state()

    filename = f"data_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5"
    filepath = os.path.join(h5_folder, filename)

    if os.path.exists(filepath):
        print(f"File already exists: {filepath}")
        return filepath

    with h5py.File(filepath, 'w') as hf:
        initial_shape = (100,)
        max_shape = (None,)
        dataset = hf.create_dataset('data', shape=initial_shape, maxshape=max_shape, dtype=h5py.special_dtype(vlen=np.dtype('float32')))

        total_steps = 0
        while True:
            episode_data = env.run_simulation(max_steps=steps_per_episode)
            for step_data in episode_data:
                for obs in step_data.values():
                    if total_steps >= dataset.shape[0]:
                        dataset.resize((dataset.shape[0] * 2,))

                    dataset[total_steps] = np.void(np.array(obs))
                    total_steps += 1

                    if total_steps >= 10000:  # Limit to 10000 steps per configuration
                        break
                
                if total_steps >= 10000:
                    break
            
            if total_steps >= 10000:
                break

        dataset.resize((total_steps,))

    print(f"Data collection complete. File saved: {filepath}")
    return filepath

def process_config(args):
    config, config_path, h5_folder = args
    return collect_data_for_config(config, config_path, steps_per_episode=100, h5_folder=h5_folder)

def load_progress():
    progress_file = os.path.join(H5_FOLDER, H5_PROGRESS_FILE)
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(f.read().splitlines())
    return set()

def save_progress(completed_configs):
    with open(os.path.join(H5_FOLDER, H5_PROGRESS_FILE), 'w') as f:
        for config in completed_configs:
            f.write(f"{config}\n")

def save_training_state(autoencoder, epoch):
    state = {
        'model_state_dicts': [ae.state_dict() for ae in autoencoder.autoencoders],
        'optimizer_state_dicts': [opt.state_dict() for opt in autoencoder.optimizers],
        'epoch': epoch
    }
    torch.save(state, os.path.join(H5_FOLDER, TRAINING_STATE_FILE))

def load_training_state(autoencoder):
    state_path = os.path.join(H5_FOLDER, TRAINING_STATE_FILE)
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=autoencoder.device)
        for i, ae in enumerate(autoencoder.autoencoders):
            ae.load_state_dict(state['model_state_dicts'][i])
            autoencoder.optimizers[i].load_state_dict(state['optimizer_state_dicts'][i])
        return state['epoch']
    return 0

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    original_config = load_config(config_path)
    
    # Set up ranges for randomization
    seed_range = range(1,11)
    num_agents_range = range(1, 6)
    num_targets_range = range(1, 6)
    num_jammers_range = range(0, 4)
    
    completed_configs = load_progress()
    configs = [
        {'seed': seed, 'n_agents': num_agents, 'n_targets': num_targets, 'n_jammers': num_jammers}
        for seed in seed_range
        for num_agents in num_agents_range
        for num_targets in num_targets_range
        for num_jammers in num_jammers_range
    ]
    
    configs_to_process = [
        (config, config_path, H5_FOLDER)
        for config in configs
        if f"data_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5" not in completed_configs
    ]
    
    # Use half of the available CPU cores for data collection
    num_processes = max(1, mp.cpu_count() // 2)
    with mp.Pool(processes=num_processes) as pool:
        for filepath in tqdm(pool.imap_unordered(process_config, configs_to_process), total=len(configs_to_process)):
            completed_configs.add(os.path.basename(filepath))
            save_progress(completed_configs)
    
    print("All configurations processed. Starting autoencoder training...")

    h5_files = [os.path.join(H5_FOLDER, filename) for filename in completed_configs]
    
    # Initialize autoencoder with the shape of the first batch
    with h5py.File(h5_files[0], 'r') as f:
        sample_data = f['data'][0]
        input_shape = sample_data['full_state'].shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = EnvironmentAutoencoder(input_shape, device)
    start_epoch = load_training_state(autoencoder)

    dataset = H5Dataset(h5_files)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    num_epochs = 100
    for epoch in range(start_epoch, num_epochs):
        loss = autoencoder.train(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {loss:.4f}")
        
        save_training_state(autoencoder, epoch + 1)
        
        if (epoch + 1) % 10 == 0:
            autoencoder.save(os.path.join(H5_FOLDER, f"autoencoder_epoch_{epoch+1}.pth"))

    autoencoder.save(os.path.join(H5_FOLDER, AUTOENCODER_FILE))
    print(f"Autoencoder training completed and model saved at {os.path.join(H5_FOLDER, AUTOENCODER_FILE)}")

if __name__ == "__main__":
    main()