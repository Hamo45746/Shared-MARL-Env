import sys
import os
from rllib_env import Environment

config_path = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/config.yaml' 
#checkpoint_dir = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/custom_ray_results/'
checkpoint_dir = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/outputs/2of10 only explore and saftey/custom_ray_results/'
params_path = '/Users/alexandramartinwallace/ray_results/PPO_custom_multi_agent_env_2024-09-30_14-13-08y9z1ytfq/params.pkl'
env2 = Environment(config_path)
env2.run_simulation_with_policy(checkpoint_dir=checkpoint_dir, params_path=params_path, max_steps=900)