import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
import yaml
import numpy as np
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from ray.rllib.algorithms.callbacks import DefaultCallbacks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
from gymnasium import spaces
import glob
import matplotlib.pyplot as plt
from rllib_env import Environment

class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Retrieve custom metrics from the environment
        env = base_env.get_sub_environments()[0]
        unique_targets_seen = env.last_seen_targets
        map_explored_percentage = env.last_map_explored_percentage
        
        # Store custom metrics
        episode.custom_metrics["unique_targets_seen"] = unique_targets_seen
        episode.custom_metrics["map_explored_percentage"] = map_explored_percentage

def env_creator(env_config):
    return Environment(config_path=env_config["config_path"], render_mode=env_config.get("render_mode", "human"))

register_env("custom_multi_agent_env", env_creator)

# Define the policy mapping function - This is for centralised training
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_0"

# Load the configuration file
with open("marl_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Path to the folder you want to continue logging to
logdir = "./custom_ray_results"
# Continue logging to the same directory
config["local_dir"] = logdir
config["callbacks"] = CustomMetricsCallback

# Update the policies in the config
num_agents = 5
obs_shape = (4, 32)  # Adjust based on your encoded observation space
action_space = spaces.Box(low=-5, high=5, shape=(2,), dtype=np.float32)
obs_space = spaces.Dict({
    "encoded_map": spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32),
    "velocity": spaces.Box(low=-30.0, high=30.0, shape=(2,), dtype=np.float32),
    "goal": spaces.Box(low=-2000, high=2000, shape=(2,), dtype=np.float32)
})

config["multiagent"]["policies"] = {
    "policy_0": (None, obs_space, action_space, {})
}
config["multiagent"]["policy_mapping_fn"] = policy_mapping_fn

# Initialize the PPO trainer
trainer = PPO(config=config)

# Lists to store metrics for plotting
targets_seen_over_time_min = []
targets_seen_over_time_mean = []
targets_seen_over_time_max = []
map_explored_over_time_min = []
map_explored_over_time_mean = []
map_explored_over_time_max = []

# Path to the latest policy checkpoint in the custom_ray_results folder
checkpoint_dir = "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/custom_ray_results"

# Restore from the checkpoint if it exists
if os.path.exists(checkpoint_dir):
    print(f"Restoring from checkpoint: {checkpoint_dir}")
    trainer.restore(checkpoint_dir)

# Resume Training from the last checkpoint
for i in range(300, 450):  # Continue training from iteration 200 onwards
    print(f"Training iteration {i}")
    result = trainer.train()

    # Access the last episode's custom metrics
    targets_found_min = result['env_runners']['custom_metrics'].get("unique_targets_seen_min")
    targets_found_mean = result['env_runners']['custom_metrics'].get("unique_targets_seen_mean")
    targets_found_max = result['env_runners']['custom_metrics'].get("unique_targets_seen_max")
    map_explored_percentage_min = result['env_runners']['custom_metrics'].get("map_explored_percentage_min")
    map_explored_percentage_mean = result['env_runners']['custom_metrics'].get("map_explored_percentage_mean")
    map_explored_percentage_max = result['env_runners']['custom_metrics'].get("map_explored_percentage_max")

    # Append metrics to lists for plotting
    targets_seen_over_time_min.append(targets_found_min)
    targets_seen_over_time_mean.append(targets_found_mean)
    targets_seen_over_time_max.append(targets_found_max)
    map_explored_over_time_min.append(map_explored_percentage_min)
    map_explored_over_time_mean.append(map_explored_percentage_mean)
    map_explored_over_time_max.append(map_explored_percentage_max)

    print(f"Iteration: {i}, "
          f"Episode Reward Mean: {result['env_runners']['episode_reward_mean']}, "
          f"Episode Length Mean: {result['env_runners']['episode_len_mean']}, "
          f"Policy Loss: {result['info']['learner']['policy_0']['learner_stats']['policy_loss']}, "
          f"Entropy: {result['info']['learner']['policy_0']['learner_stats']['entropy']},"
          f"Unique targets seen min: {targets_found_min},"
          f"Unique targets seen mean: {targets_found_mean},"
          f"Unique targets seen max: {targets_found_max},"
          f"Map Explored Percentage min: {map_explored_percentage_min:.2f}%,"
          f"Map Explored Percentage mean: {map_explored_percentage_mean:.2f}%,"
          f"Map Explored Percentage max: {map_explored_percentage_max:.2f}%")

    # Save the checkpoint every 20 iterations
    if (i + 1) % 20 == 0:
        checkpoint_dir = trainer.save("/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/custom_ray_results")

    if (i + 1) % 50 == 0:
        checkpoint_dir = trainer.save(logdir)
        latest_folder = max(glob.glob(os.path.join('/Users/alexandramartinwallace/ray_results/', '*/')), key=os.path.getmtime)
        params_path = os.path.join(latest_folder, "params.pkl")
        env2 = Environment(config_path="config.yaml")
        env2.run_simulation_with_policy(checkpoint_dir=checkpoint_dir, params_path=params_path, max_steps=100, iteration=i)
# Ensure TensorBoard logs are being written
final_checkpoint_dir = trainer.save("/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/custom_ray_results")

# Plot the metrics
plt.figure(figsize=(12, 5))

# Plot unique targets seen
plt.subplot(1, 2, 1)
plt.plot(targets_seen_over_time_min, label='Unique Targets Seen_min')
plt.plot(targets_seen_over_time_mean, label='Unique Targets Seen_mean')
plt.plot(targets_seen_over_time_max, label='Unique Targets Seen_max')
plt.xlabel('Iteration')
plt.ylabel('Unique Targets Seen')
plt.title('Unique Targets Seen Over Training')
plt.legend()

# Plot map explored percentage
plt.subplot(1, 2, 2)
plt.plot(map_explored_over_time_min, label='Map Explored Percentage_min')
plt.plot(map_explored_over_time_mean, label='Map Explored Percentage_mean')
plt.plot(map_explored_over_time_max, label='Map Explored Percentage_max')
plt.xlabel('Iteration')
plt.ylabel('Map Explored (%)')
plt.title('Map Explored Percentage Over Training')
plt.legend()

# Save the plot to a file
plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()
plt.close()

