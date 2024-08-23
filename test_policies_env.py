# This file is to test the policies in the simulation 
import sys
import os
from rllib_env import Environment

config_path = 'config.yaml' 
env = Environment(config_path)
env.run_simulation_with_policy(checkpoint_path="custom_ray_results/rllib_checkpoint.json", max_steps=100)