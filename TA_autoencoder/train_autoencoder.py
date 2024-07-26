import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
import multiprocessing as mp
from tqdm import tqdm
import logging
import psutil
import time
import gc
import signal
from memory_profiler import profile

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from autoencoder import EnvironmentAutoencoder

# Constants
H5_FOLDER = '/Volumes/T7 Shield/METR4911/Mem_profiling_test'
# H5_FOLDER = '/media/rppl/T7 Shield/METR4911/Mem_profiling_test'
H5_PROGRESS_FILE = 'h5_collection_progress.txt'
AUTOENCODER_FILE = 'trained_autoencoder.pth'
TRAINING_STATE_FILE = 'training_state.pth'
STEPS_PER_EPISODE = 100

logging.basicConfig(filename='autoencoder_training.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

global_pool = None

class FlattenedMultiAgentH5Dataset(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files
        self.data_index = self._index_data()

    def _index_data(self):
        index = []
        for file_idx, h5_file in enumerate(self.h5_files):
            with h5py.File(h5_file, 'r') as f:
                for step_idx in f['data']:
                    step_data = f['data'][step_idx]
                    for agent_id in step_data:
                        index.append((file_idx, step_idx, agent_id))
        return index

    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, idx):
        file_idx, step_idx, agent_id = self.data_index[idx]
        with h5py.File(self.h5_files[file_idx], 'r') as f:
            agent_data = f['data'][step_idx][agent_id]
            full_state = agent_data['full_state'][()]

        return {f'layer_{i}': torch.FloatTensor(full_state[i]).unsqueeze(0) for i in range(full_state.shape[0])}

def cleanup_resources():
    global global_pool
    if global_pool:
        global_pool.close()
        global_pool.join()
    
    for child in mp.active_children():
        child.terminate()
        child.join(timeout=1)
    
    mp.util.cleanup_remaining_resources()

def signal_handler(signum, frame):
    print("Interrupted. Cleaning up resources...")
    cleanup_resources()
    sys.exit(1)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def is_dataset_complete(filepath, steps_per_episode):
    try:
        with h5py.File(filepath, 'r') as hf:
            return 'data' in hf and len(hf['data']) == steps_per_episode
    except Exception as e:
        logging.error(f"Error checking dataset completeness for {filepath}: {str(e)}")
        return False

@profile
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

    # Check if file exists and determine the last completed step
    start_step = 0
    if os.path.exists(filepath):
        with h5py.File(filepath, 'r') as hf:
            if 'data' in hf:
                existing_steps = [int(step) for step in hf['data'].keys()]
                if existing_steps:
                    start_step = max(existing_steps) + 1
                    logging.info(f"Resuming data collection for {filename} from step {start_step}")
                    
                    # If all steps are completed, return the filepath
                    if start_step >= steps_per_episode:
                        logging.info(f"All steps already collected for {filename}")
                        return filepath

    # Open the file in append mode if it exists, otherwise create a new file
    with h5py.File(filepath, 'a') as hf:
        if 'data' not in hf:
            dataset = hf.create_group('data', track_order=True)
        else:
            dataset = hf['data']
        
        # Set the environment to the correct state if resuming
        if start_step > 0:
            for _ in range(start_step):
                action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(env.agents)}
                env.step(action_dict)

        for step in tqdm(range(start_step, steps_per_episode), desc=f"Collecting data for {filename}", initial=start_step, total=steps_per_episode):
            action_dict = {agent_id: agent.get_next_action() for agent_id, agent in enumerate(env.agents)}
            observations, rewards, done, info = env.step(action_dict)
            
            if not isinstance(observations, dict):
                logging.error(f"Unexpected observation type: {type(observations)}")
                break

            step_group = dataset.create_group(str(step), track_order=True)
            for agent_id, obs in observations.items():
                agent_group = step_group.create_group(str(agent_id), track_order=True)
                
                # Use compression for datasets
                agent_group.create_dataset('full_state', data=obs['full_state'], 
                                           compression="gzip", compression_opts=9)
                agent_group.create_dataset('local_obs', data=obs['local_obs'], 
                                           compression="gzip", compression_opts=9)
            
            if step % 10 == 0:
                hf.flush()
                gc.collect()
                mem_usage = psutil.virtual_memory().percent
                logging.info(f"Step {step}: Memory usage {mem_usage}%")
                
                if mem_usage > 90:
                    logging.warning(f"High memory usage detected: {mem_usage}%")
                    break
            
            if done:
                break

    logging.info(f"Data collection complete. File saved: {filepath}")
    return filepath

def process_config(args):
    try:
        config, config_path, h5_folder, steps_per_episode = args
        filepath = collect_data_for_config(config, config_path, steps_per_episode, h5_folder)
        if is_dataset_complete(filepath, steps_per_episode):
            return filepath
        else:
            return None
    except Exception as e:
        logging.error(f"Error processing config {config}: {str(e)}")
        return None

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

def train_autoencoder(autoencoder, h5_files, num_epochs=100, batch_size=32, start_epoch=0):
    dataset = FlattenedMultiAgentH5Dataset(h5_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(start_epoch, num_epochs):
        try:
            loss = autoencoder.train(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {loss:.4f}")
            logging.info(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {loss:.4f}")
            
            save_training_state(autoencoder, epoch + 1)
            
            if (epoch + 1) % 10 == 0:
                autoencoder.save(os.path.join(H5_FOLDER, f"autoencoder_epoch_{epoch+1}.pth"))
            
            # Monitor system resources
            mem_percent = psutil.virtual_memory().percent
            logging.info(f"Memory usage: {mem_percent}%")
            
            if mem_percent > 90:
                logging.warning("High memory usage detected. Pausing for 60 seconds.")
                time.sleep(60)  # Give system time to free up memory

        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    autoencoder.save(os.path.join(H5_FOLDER, AUTOENCODER_FILE))
    logging.info(f"Autoencoder training completed and model saved at {os.path.join(H5_FOLDER, AUTOENCODER_FILE)}")

@profile
def main():
    global global_pool
    
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
    
    configs_to_process = []
    for config in configs:
        filename = f"data_s{config['seed']}_t{config['n_targets']}_j{config['n_jammers']}_a{config['n_agents']}.h5"
        filepath = os.path.join(H5_FOLDER, filename)
        if not is_dataset_complete(filepath, STEPS_PER_EPISODE):
            configs_to_process.append((config, config_path, H5_FOLDER, STEPS_PER_EPISODE))
    
    # Use all but 1 of the available CPU cores for data collection
    num_processes = max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=num_processes) as pool:
        global_pool = pool
        for filepath in tqdm(pool.imap_unordered(process_config, configs_to_process), total=len(configs_to_process)):
            if filepath and is_dataset_complete(filepath, STEPS_PER_EPISODE):
                completed_configs.add(os.path.basename(filepath))
                save_progress(completed_configs)
            
            # Check overall memory usage
            mem_usage = psutil.virtual_memory().percent
            if mem_usage > 90:
                logging.warning(f"High overall memory usage detected: {mem_usage}%. Pausing for 60 seconds.")
                time.sleep(60)  # Give system time to free up memory
    
    global_pool = None  # Reset the global pool after it's closed
    
    print("All configurations processed. Starting autoencoder training...")

    h5_files = [os.path.join(H5_FOLDER, filename) for filename in completed_configs]
    
    # Initialize autoencoder with the shape of the first batch
    with h5py.File(h5_files[0], 'r') as f:
        first_step = f['data']['0']
        first_agent = first_step[list(first_step.keys())[0]]
        input_shape = first_agent['full_state'].shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = EnvironmentAutoencoder(input_shape, device)
    start_epoch = load_training_state(autoencoder)

    train_autoencoder(autoencoder, h5_files, start_epoch=start_epoch)

if __name__ == "__main__":
    # Set up the signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        mp.set_start_method('spawn', force=True)
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        cleanup_resources()
        print("All resources cleaned up")