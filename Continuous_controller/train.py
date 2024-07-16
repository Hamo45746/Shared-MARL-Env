import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
from rllib_env import Environment
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from gymnasium import spaces


# Register the custom environment
def env_creator(env_config):
    return Environment(config_path=env_config["config_path"], render_mode=env_config.get("render_mode", "human"))

register_env("custom_multi_agent_env", env_creator)

# Define policies and policy mapping function
num_agents = 2  # Example, adjust based on your environment
obs_range = 17  # Example, adjust based on your observation range

policies = {
    "policy_0": (None, spaces.Dict({
        agent_id: spaces.Box(low=-20, high=1, shape=(4, obs_range, obs_range), dtype=np.float32)
        for agent_id in range(num_agents)
    }), spaces.Discrete(5), {}),  # Adjust the action space accordingly
}

def policy_mapping_fn(agent_id):
    return "policy_0"

# Configure the PPO trainer
config = {
    "env": "custom_multi_agent_env",
    "env_config": {
        "config_path": "config.yaml",  # Provide the path to your config
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn": policy_mapping_fn,
    },
    "num_workers": 1,
    "framework": "torch",  # or "tf"
    "lr": 0.0003,
    "train_batch_size": 4000,
    "sgd_minibatch_size": 64,
    "num_sgd_iter": 10,
    "rollout_fragment_length": 200,
    "batch_mode": "truncate_episodes",
    "num_gpus": 0,
}

trainer = PPO(config=config)

# Train the agents
for i in range(100):
    result = trainer.train()
    print(f"Iteration: {i}, Reward: {result['episode_reward_mean']}")





