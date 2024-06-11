import numpy as np
import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
#from stable_baselines3 import PPO
#from stable_baselines3.common.env_checker import check_env
from env import Environment

# def main(config_path, timesteps=10000, save_path="ppo_agent"):
#     # Initialize the environment
#     env = Environment(config_path)

#     # Ensure the environment is compatible with Gymnasium
#     check_env(env, warn=True)

#     # Initialize the RL model (PPO)
#     model = PPO("MlpPolicy", env, verbose=1)

#     # Train the model
#     model.learn(total_timesteps=timesteps)

#     # Save the model
#     model.save(save_path)

# if __name__ == "__main__":
#     print("in here")
#     #config_path = 'config.yaml'  # Path to the configuration file
#     #main(config_path, timesteps=10000, save_path="ppo_agent")

config_path = 'config.yaml' 
env = Environment(config_path)
Environment.run_simulation(env)