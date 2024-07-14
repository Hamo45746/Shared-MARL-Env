import sys
import os
import multiprocessing as mp

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from autoencoder import EnvironmentAutoencoder
import numpy as np
import torch
import yaml

DATA_FOLDER = 'collected_data'
COMBINED_DATA_FILE = 'combined_data.npy'
PROGRESS_FILE = 'collection_progress.txt'

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def collect_data_for_config(config, config_path, steps_per_episode):
    env = Environment(config_path)
    env.config.update(config)
    env.num_agents = config['n_agents']
    env.initialise_agents()
    env.initialise_targets()
    env.initialise_jammers()
    env.update_global_state()
    
    data = []
    episode_data = env.run_simulation(max_steps=steps_per_episode)
    for step_data in episode_data:
        for obs in step_data.values():
            if isinstance(obs, dict) and 'full_state' in obs:
                data.append(obs['full_state'])
            else:
                data.append(obs)
    return np.array(data)

def get_config_filename(seed, num_targets, num_jammers, num_agents):
    return f"data_s{seed}_t{num_targets}_j{num_jammers}_a{num_agents}.npy"

def save_config_data(data, filename):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    np.save(os.path.join(DATA_FOLDER, filename), data)

def load_config_data(filename):
    filepath = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(filepath):
        return np.load(filepath)
    return None

def process_config(args):
    seed, num_targets, num_jammers, num_agents, original_config, config_path = args
    config_filename = get_config_filename(seed, num_targets, num_jammers, num_agents)
    config_data = load_config_data(config_filename)
    
    if config_data is not None:
        print(f"Loaded data for seed={seed}, targets={num_targets}, jammers={num_jammers}, agents={num_agents}")
        return config_filename
    
    print(f"Collecting data for seed={seed}, targets={num_targets}, jammers={num_jammers}, agents={num_agents}")
    config = original_config.copy()
    config['seed'] = seed
    config['n_targets'] = num_targets
    config['n_jammers'] = num_jammers
    config['n_agents'] = num_agents
    
    data = collect_data_for_config(config, config_path, steps_per_episode=100)
    save_config_data(data, config_filename)
    return config_filename

def save_progress(completed_configs):
    with open(os.path.join(DATA_FOLDER, PROGRESS_FILE), 'w') as f:
        for config in completed_configs:
            f.write(f"{config}\n")

def load_progress():
    progress_file = os.path.join(DATA_FOLDER, PROGRESS_FILE)
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return set(f.read().splitlines())
    return set()

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    original_config = load_config(config_path)
    
    # Set up ranges for randomization
    seed_range = range(1, 11)
    num_targets_range = range(1, 6)
    num_jammers_range = range(0, 4)
    num_agents_range = range(1, 6)
    
    completed_configs = load_progress()
    configs = [
        (seed, num_targets, num_jammers, num_agents, original_config, config_path)
        for seed in seed_range
        for num_targets in num_targets_range
        for num_jammers in num_jammers_range
        for num_agents in num_agents_range
        if get_config_filename(seed, num_targets, num_jammers, num_agents) not in completed_configs
    ]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for config_filename in pool.imap_unordered(process_config, configs):
            completed_configs.add(config_filename)
            save_progress(completed_configs)
    
    print("All configurations processed. Combining data...")
    all_data = [load_config_data(filename) for filename in completed_configs]
    combined_data = np.concatenate(all_data)
    np.save(os.path.join(DATA_FOLDER, COMBINED_DATA_FILE), combined_data)
    
    print("Creating and training autoencoder...")
    input_shape = combined_data.shape[1:]
    autoencoder = EnvironmentAutoencoder(input_shape)
    
    combined_data = torch.FloatTensor(combined_data)
    autoencoder.train(combined_data, epochs=100, batch_size=32)
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_autoencoder.pth')
    autoencoder.save(save_path)
    
    print(f"Autoencoder training completed and model saved at {save_path}")

if __name__ == "__main__":
    main()