import sys
import os
from rllib_env import Environment

# Correct paths
config_path = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/config.yaml' 
checkpoint_dir = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/custom_ray_results/'
params_path = '/Users/alexandramartinwallace/ray_results/PPO_custom_multi_agent_env_2024-08-20_14-12-2972vkg8qg/params.pkl'

env = Environment(config_path)
env.run_simulation_with_policy(checkpoint_dir=checkpoint_dir, params_path=params_path, max_steps=300)