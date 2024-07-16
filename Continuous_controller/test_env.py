# This file is to test / run the environment without any learning  
import numpy as np
import gymnasium as gym
import sys
import os
from gymnasium.utils.env_checker import check_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
from rllib_env import Environment

config_path = 'config.yaml' 
env = Environment(config_path)
#check_env(env, warn=True) #this ruins the seed and therefor the sim won't be randomised differently evey time 
Environment.run_simulation(env)