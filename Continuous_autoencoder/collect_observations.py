import numpy as np
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rllib_env import Environment

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

    data = {"map_view": [], "agent": [], "target": [], "jammer": []}
    episode_data = env.run_simulation(max_steps=steps_per_episode)
    for step_data in episode_data:
        for obs in step_data.values():
            if isinstance(obs, dict) and 'map' in obs:
                data["map_view"].append(obs['map'][0])  # 'map' is the first layer
                data["agent"].append(obs['map'][1])  # 'agent' is the second layer
                data["target"].append(obs['map'][2])  # 'target' is the third layer
                data["jammer"].append(obs['map'][3])  #'jammer' is the fourth layer
    return {key: np.array(value) for key, value in data.items()}

def generate_random_configs(base_config, num_configs):
    configs = []
    for _ in range(num_configs):
        config = base_config.copy()
        config['seed'] = np.random.randint(0, 100)
        config['n_targets'] = np.random.randint(85, 95)  # Random number of targets > 5
        config['n_jammers'] = np.random.randint(100, 110)  # Random number of jammers > 3
        configs.append(config)
    return configs

def main():
    config_path = 'config.yaml'  # Update this to your config file path
    base_config = load_config(config_path)
    
    num_configs = 6 # Number of different settings you want to generate data for
    steps_per_episode = 300  # Number of steps per episode
    all_data_layers = {key: [] for key in ["map_view", "agent", "target", "jammer"]}
    random_configs = generate_random_configs(base_config, num_configs)
    for config in random_configs:
        data_layers = collect_data_for_config(config, config_path, steps_per_episode)
        for key in all_data_layers.keys():
            all_data_layers[key].append(data_layers[key])

    for key in all_data_layers.keys():
        all_data_layers[key] = np.concatenate(all_data_layers[key], axis=0)
        np.save(f'outputs/combined_data_{key}.npy', all_data_layers[key])

if __name__ == "__main__":
    main()