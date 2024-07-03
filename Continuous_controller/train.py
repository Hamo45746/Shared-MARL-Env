# This file is to try and train the agents using the stable_baselines3 PPO method 
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import gymnasium as gym
import sys
import os
from gymnasium.utils.env_checker import check_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
from stable_baselines3.common.env_util import make_vec_env
from env import Environment
from stable_baselines3 import PPO
#from stable_baselines3.common.env_checker import check_env

def main(config_path, timesteps=1, save_path="ppo_agent"):
    # Initialize the environment
    env = Environment(config_path)

    env = make_vec_env(lambda: env, n_envs=1)

    # Initialize the RL model (PPO)
    model = PPO("MultiInputPolicy", env, verbose=1)

    # Train the model
    model.learn(total_timesteps=timesteps)

    # Save the model
    model.save(save_path)

if __name__ == "__main__":
    config_path = 'config.yaml'  # Path to the configuration file
    main(config_path, timesteps=1, save_path="ppo_agent")






