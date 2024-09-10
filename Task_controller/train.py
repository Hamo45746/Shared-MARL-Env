import os
import sys
import yaml
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from gymnasium import spaces
from ray.tune.logger import DEFAULT_LOGGERS

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from rllib_wrapper import RLLibEnvWrapper

# Environment creator function
def env_creator(env_config):
    env = Environment(config_path=env_config["config_path"])
    return RLLibEnvWrapper(env, ae_folder_path=env_config["ae_folder_path"])

# Register the environment
register_env("custom_multi_agent_env", env_creator)

# Load the configuration file
with open("marl_config.yaml", "r") as file:
    config = yaml.safe_load(file)
    
# Update the configuration
config.update({
    "num_workers": 2,
    "num_envs_per_worker": 1,
    "train_batch_size": 10000,  # Increased batch size
    "rollout_fragment_length": 1000,  # Increased fragment length
    "sgd_minibatch_size": 256,
    "num_sgd_iter": 30,
    "framework": "torch",
    "log_level": "DEBUG",
    "batch_mode": "truncate_episodes",  # Allow learning from partial episodes
    "sample_async": True,  # Enable asynchronous sampling
})

# Define the policy mapping function for decentralized training
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return f"policy_{agent_id}"

# Set up custom location for ray_results
logdir = "./custom_ray_results"
config["local_dir"] = logdir

# Update the policies in the config
num_agents = 10  # Adjust based on your environment
obs_shape = (5, 256)  # Adjusted for encoded observation space (4 layers + 1 battery layer, 256 encoding size)
action_space = spaces.Discrete((2 * 15 + 1) ** 2)  # Assuming max_steps_per_action is 15
obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

# Config for decentralized training
config["multiagent"] = {
    "policies": {f"policy_{i}": (None, obs_space, action_space, {}) for i in range(num_agents)},
    "policy_mapping_fn": policy_mapping_fn
}

# Update environment config
config["env_config"] = {
    "config_path": "config.yaml",
    "ae_folder_path": "/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data/AE_save_06_09"
}

# Set up logging
config["logger_config"] = {
    "type": "ray.tune.logger.UnifiedLogger",
    "logdir": logdir,
    "loggers": DEFAULT_LOGGERS
}

ray.init(num_gpus=1, logging_level=ray.logging.DEBUG)

# Initialise the PPO trainer
trainer = PPO(config=config)

for i in range(50):
    print(f"Iteration: {i}")
    result = trainer.train()
    
    # Print training metrics
    # print(f"Iteration {i} result keys: {result.keys()}")
    if 'episode_reward_mean' in result:
        print(f"Episode Reward Mean: {result['episode_reward_mean']}")
    if 'episode_len_mean' in result:
        print(f"Episode Length Mean: {result['episode_len_mean']}")
    
    # Print more detailed metrics
    print(f"Total time trained: {result['time_total_s']}")
    print(f"Timesteps trained: {result['timesteps_total']}")
    print(f"Episodes this iteration: {result.get('episodes_this_iter', 0)}")
    
    # Save checkpoint every 10 iterations
    if (i + 1) % 10 == 0:
        checkpoint_dir = trainer.save(logdir)
        print(f"Checkpoint saved at {checkpoint_dir}")

        # Optional: Run a simulation with the saved policy
        # env = Environment(config_path="config.yaml")
        # env.run_simulation_with_policy(checkpoint_dir=checkpoint_dir, max_steps=100, iteration=i)

# Save final checkpoint
final_checkpoint_dir = trainer.save(logdir)
print(f"Final checkpoint saved at {final_checkpoint_dir}")

ray.shutdown()