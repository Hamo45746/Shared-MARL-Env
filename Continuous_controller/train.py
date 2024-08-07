import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import gymnasium as gym
import sys
import yaml
from ray.tune.logger import UnifiedLogger, TBXLoggerCallback
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
from rllib_env import Environment
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from gymnasium import spaces


def env_creator(env_config):
    return Environment(config_path=env_config["config_path"], render_mode=env_config.get("render_mode", "human"))

register_env("custom_multi_agent_env", env_creator)

# Load the configuration file
with open("marl_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Define the policy mapping function
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_0"

# Update the policies in the config
num_agents = 5  # Example, adjust based on your environment
obs_shape = (32, 4) #NEED TO ADJUST TO ENCODED OBSERVATION SPACE
action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
obs_space = spaces.Dict({
    "encoded_map": spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32),
    "velocity": spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
    "goal": spaces.Box(low=-2000, high=2000, shape=(2,), dtype=np.float32)
})
config["multiagent"]["policies"] = {
    "policy_0": (None, obs_space, action_space, {})
}
config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

# Add TensorBoard logger callback
logdir = "./logs"
config["logger_config"] = {
    "loggers": [UnifiedLogger, TBXLoggerCallback],
    "logdir": logdir
}

# Initialize the PPO trainer
trainer = PPO(config=config)

# Train the agents
for i in range(100):
    print(i)
    result = trainer.train()
    print(f"Iteration: {i}, Results: {result}")
    # Check if 'episode_reward_mean' is in the result
    if 'episode_reward_mean' in result:
        print(f"Iteration: {i}, Reward: {result['episode_reward_mean']}")
    else:
        print(f"Iteration: {i}, 'episode_reward_mean' not found in result")

# Ensure TensorBoard logs are being written
trainer.save(logdir)