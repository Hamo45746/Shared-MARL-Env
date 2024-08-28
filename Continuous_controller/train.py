import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
import numpy as np
import sys
import yaml
from ray.tune.logger import TBXLoggerCallback, JsonLoggerCallback, CSVLoggerCallback
from ray.tune.logger import TBXLogger
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
from rllib_env import Environment
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from gymnasium import spaces
import glob
import os


def env_creator(env_config):
    return Environment(config_path=env_config["config_path"], render_mode=env_config.get("render_mode", "human"))

register_env("custom_multi_agent_env", env_creator)

# Load the configuration file
with open("marl_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# # Define the policy mapping function - This is for centralised training 
# def policy_mapping_fn(agent_id, episode, worker, **kwargs):
#     return "policy_0"

# Define the policy mapping function - this is for decentralised training
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"policy_{agent_id}"

# Update the configuration
logdir = "./custom_ray_results"

# Update the policies in the config
num_agents = 5  # Example, adjust based on your environment
obs_shape = (4, 32) #NEED TO ADJUST TO ENCODED OBSERVATION SPACE
action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
obs_space = spaces.Dict({
    "encoded_map": spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32),
    "velocity": spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32),
    "goal": spaces.Box(low=-2000, high=2000, shape=(2,), dtype=np.float32)
})
# # This is the config for cnetralised training - one policy
# config["multiagent"]["policies"] = {
#     "policy_0": (None, obs_space, action_space, {})
# }
# config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

# This is the config for decentralised training - multiple policies. 
config["multiagent"]["policies"] = {
    f"policy_{i}": (None, obs_space, action_space, {}) for i in range(num_agents)
}
config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn


# Initialize the PPO trainer
trainer = PPO(config=config)

# Train the agents
for i in range(200):
    print(i)
    result = trainer.train()
    print(f"Iteration: {i}, "
          f"Episode Reward Mean: {result['env_runners']['episode_reward_mean']}, "
          f"Episode Length Mean: {result['env_runners']['episode_len_mean']}, "
          f"Policy Loss: {result['info']['learner']['policy_0']['learner_stats']['policy_loss']}, "
          f"Entropy: {result['info']['learner']['policy_0']['learner_stats']['entropy']}")
    
    if (i + 1) % 50 == 0:
        checkpoint_dir = trainer.save(logdir)
        latest_folder = max(glob.glob(os.path.join('/Users/alexandramartinwallace/ray_results/', '*/')), key=os.path.getmtime)
        params_path = os.path.join(latest_folder, "params.pkl")
        env2 = Environment(config_path="config.yaml")
        env2.run_simulation_with_policy(checkpoint_dir=checkpoint_dir, params_path=params_path, max_steps=100, iteration=i)

#Ensure TensorBoard logs are being written
final_checkpoint_dir = trainer.save(logdir)