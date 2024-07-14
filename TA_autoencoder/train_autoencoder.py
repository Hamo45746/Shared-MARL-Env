import os
import numpy as np
import torch
import yaml
import pickle
from env import Environment
from autoencoder import EnvironmentAutoencoder

DATA_FOLDER = 'collected_data'
COMBINED_DATA_FILE = 'combined_data.npy'

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def collect_data(env, num_episodes, steps_per_episode):
    data = []
    for _ in range(num_episodes):
        episode_data = env.run_simulation(max_steps=steps_per_episode)
        for step_data in episode_data:
            for agent_id, obs in step_data.items():
                if isinstance(obs, dict) and 'full_state' in obs:
                    data.append(obs['full_state'])
                else:
                    data.append(obs)
    return np.array(data)

def get_config_filename(seed, num_targets, num_jammers, num_agents):
    return f"data_s{seed}_t{num_targets}_j{num_jammers}_a{num_agents}.pkl"

def save_config_data(data, filename):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    with open(os.path.join(DATA_FOLDER, filename), 'wb') as f:
        pickle.dump(data, f)

def load_config_data(filename):
    filepath = os.path.join(DATA_FOLDER, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
    original_config = load_config(config_path)
    
    combined_data_path = os.path.join(DATA_FOLDER, COMBINED_DATA_FILE)
    
    if os.path.exists(combined_data_path):
        print("Loading existing combined data...")
        combined_data = np.load(combined_data_path)
    else:
        print("Collecting new data...")
        # Set up ranges for randomization
        seed_range = range(1, 11)  # 10 different seeds
        num_targets_range = range(1, 11)  # 1 to 10 targets
        num_jammers_range = range(0, 6)  # 0 to 5 jammers
        num_agents_range = range(1, 11)  # 1 to 10 agents
        
        all_data = []
        
        for seed in seed_range:
            for num_targets in num_targets_range:
                for num_jammers in num_jammers_range:
                    for num_agents in num_agents_range:
                        config_filename = get_config_filename(seed, num_targets, num_jammers, num_agents)
                        config_data = load_config_data(config_filename)
                        
                        if config_data is not None:
                            print(f"Loading existing data for seed={seed}, targets={num_targets}, jammers={num_jammers}, agents={num_agents}")
                            all_data.append(config_data)
                        else:
                            print(f"Collecting data for seed={seed}, targets={num_targets}, jammers={num_jammers}, agents={num_agents}")
                            
                            config = original_config.copy()
                            config['seed'] = seed
                            config['n_targets'] = num_targets
                            config['n_jammers'] = num_jammers
                            config['n_agents'] = num_agents
                            
                            env = Environment(config_path)
                            env.config.update(config)
                            env.num_agents = num_agents
                            env.reset()
                            
                            data = collect_data(env, num_episodes=5, steps_per_episode=100)
                            all_data.append(data)
                            save_config_data(data, config_filename)
        
        combined_data = np.concatenate(all_data, axis=0)
        np.save(combined_data_path, combined_data)
    
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