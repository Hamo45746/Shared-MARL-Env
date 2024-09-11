import os
import sys
import yaml
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from gymnasium import spaces
from ray.tune.logger import DEFAULT_LOGGERS
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import Environment
from rllib_wrapper import RLLibEnvWrapper

def env_creator(env_config):
    env = Environment(config_path=env_config["config_path"])
    return RLLibEnvWrapper(env, ae_folder_path=env_config["ae_folder_path"])

logdir = "./custom_ray_results"
# Update the policies in the config
num_agents = 10  # Adjust based on your environment
obs_shape = (5, 256)  # Adjusted for encoded observation space (4 layers + 1 battery layer, 256 encoding size)
action_space = spaces.Discrete((2 * 15 + 1) ** 2)  # Assuming max_steps_per_action is 15
obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)



config = (
    PPOConfig()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment(env_creator)
    .env_runners(num_env_runners=4)  # Adjust based on your needs
    .training(
        model={
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "tanh",
        },
        lr=1e-5,
        gamma=0.99,
        lambda_=0.95,
        clip_param=0.2,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        train_batch_size=4096,
        sgd_minibatch_size=256,
        num_sgd_iter=1,
    )
    .resources(num_gpus=1)
    .rollouts(num_rollout_workers=4)
    .debugging(log_level="DEBUG")
)

# Set up multi-agent policies
policies = {
    f"policy_{i}": PolicySpec(observation_space=obs_space, action_space=action_space)
    for i in range(num_agents)
}

config = config.multi_agent(
    policies=policies,
    policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"policy_{agent_id}",
)

# Set environment config
config.env_config = {
    "config_path": "config.yaml",
    "ae_folder_path": "/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data/AE_save_06_09"
}

# Build the algorithm
algo = config.build()

for i in range(50):
    print(f"Iteration: {i}")
    result = algo.train()
    
    # Print training metrics
    print(f"Episodes this iteration: {result.get('episodes_this_iter', 0)}")
    print(f"Timesteps this iteration: {result.get('timesteps_total', 0)}")
    print(f"Episode Reward Mean: {result.get('episode_reward_mean', 0)}")
    print(f"Episode Length Mean: {result.get('episode_len_mean', 0)}")
    
    # Save checkpoint every 10 iterations
    if (i + 1) % 10 == 0:
        checkpoint_dir = algo.save_to_path(logdir)
        print(f"Checkpoint saved at {checkpoint_dir}")

# Save final checkpoint
final_checkpoint_dir = algo.save_to_path(logdir)
print(f"Final checkpoint saved at {final_checkpoint_dir}")

algo.stop()