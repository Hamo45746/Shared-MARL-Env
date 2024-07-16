import numpy as np
import torch
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env import Environment

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

def generate_random_configs(base_config, num_configs):
    configs = []
    for _ in range(num_configs):
        config = base_config.copy()
        config['seed'] = np.random.randint(0, 100)
        config['n_targets'] = np.random.randint(5, 11)  # Random number of targets > 5
        config['n_jammers'] = np.random.randint(3, 8)  # Random number of jammers > 3
        configs.append(config)
    return configs

def main():
    config_path = 'config.yaml'  # Update this to your config file path
    base_config = load_config(config_path)
    
    num_configs = 10  # Number of different settings you want to generate data for
    steps_per_episode = 100  # Number of steps per episode
    all_data = []

    random_configs = generate_random_configs(base_config, num_configs)
    for config in random_configs:
        data = collect_data_for_config(config, config_path, steps_per_episode)
        all_data.append(data)

    all_data = np.concatenate(all_data, axis=0)
    np.save('outputs/combined_data.npy', all_data)

if __name__ == "__main__":
    main()