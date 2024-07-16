import sys
import os
import multiprocessing as mp
import numpy as np
import torch
import yaml

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from autoencoder import EnvironmentAutoencoder

# Update the data folder path
DATA_FOLDER = '/Volumes/T7 Shield/METR4911/TA_autoencoder_data'
PROGRESS_FILE = 'collection_progress.txt'
AUTOENCODER_FILE = 'trained_autoencoder.pth'
TRAINING_STATE_FILE = 'training_state.pth'

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

def save_training_state(autoencoder, epoch, optimizer):
    state = {
        'model_state_dict': autoencoder.autoencoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(state, os.path.join(DATA_FOLDER, TRAINING_STATE_FILE))

def load_training_state(autoencoder, optimizer):
    state_path = os.path.join(DATA_FOLDER, TRAINING_STATE_FILE)
    if os.path.exists(state_path):
        state = torch.load(state_path)
        autoencoder.autoencoder.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        return state['epoch']
    return 0

def train_on_batch(autoencoder, batch_data, epoch):
    batch_tensor = torch.FloatTensor(batch_data).to(autoencoder.device)
    loss = autoencoder.train(batch_tensor, start_epoch=epoch, epochs=epoch+1, batch_size=32)
    return loss

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    original_config = load_config(config_path)
    
    # Set up ranges for randomization (reverted to original)
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
    
    print("All configurations processed. Starting autoencoder training...")

    # Initialize autoencoder with the shape of the first batch
    first_batch = load_config_data(next(iter(completed_configs)))
    input_shape = first_batch.shape[1:]
    autoencoder = EnvironmentAutoencoder(input_shape)

    # Load previous training state if it exists
    start_epoch = load_training_state(autoencoder, autoencoder.optimizer)

    num_epochs = 100
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0
        batch_count = 0
        for config_filename in completed_configs:
            batch_data = load_config_data(config_filename)
            if batch_data is not None:
                loss = train_on_batch(autoencoder, batch_data, epoch)
                total_loss += loss
                batch_count += 1
                
                if batch_count % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_count}/{len(completed_configs)}, "
                          f"Average Loss: {total_loss/batch_count:.4f}")
        
        epoch_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {epoch_loss:.4f}")
        
        # Save training state
        save_training_state(autoencoder, epoch + 1, autoencoder.optimizer)
        
        # Optionally, save the model periodically
        if (epoch + 1) % 10 == 0:
            autoencoder.save(os.path.join(DATA_FOLDER, f"autoencoder_epoch_{epoch+1}.pth"))

    # Save the final model
    autoencoder.save(os.path.join(DATA_FOLDER, AUTOENCODER_FILE))
    print(f"Autoencoder training completed and model saved at {os.path.join(DATA_FOLDER, AUTOENCODER_FILE)}")

if __name__ == "__main__":
    main()