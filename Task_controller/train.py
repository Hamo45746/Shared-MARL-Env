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
from ray.rllib.env.multi_agent_env_runner import MultiAgentEnvRunner

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rllib_wrapper import RLLibEnvWrapper

def env_creator(env_config):
    from env import Environment
    env = Environment(config_path=env_config["config_path"])
    return RLLibEnvWrapper(env, ae_folder_path=env_config["ae_folder_path"])

logdir = "./custom_ray_results"
# Update the policies in the config
num_agents = 10  # Adjust based on your environment
obs_shape = (5 * 256,)  # Adjusted for encoded observation space (4 layers + 1 battery layer, 256 encoding size)
action_space = spaces.Box(low=np.array([0, 0]), high=np.array([2*40, 2*40]), dtype=np.int32)
obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5 * 256,), dtype=np.float32)

# Set up multi-agent policies
# policies = {
#     f"policy_{i}": PolicySpec(observation_space=obs_space, action_space=action_space)
#     for i in range(num_agents)
# }

#Setup one shared policy
policies = {
    f"policy_0": PolicySpec(observation_space=obs_space, action_space=action_space)
}

def policy_mapping_fn(agent_id, episode, **kwargs):
    # This function maps each agent to a policy based on its agent_id
    # return f"policy_{agent_id}" # use this for training individual policies per agent.
    return f"policy_0" # Use this for training one shared policy.

# Register the environment
register_env("custom_multi_agent_env", env_creator)

config = (
    PPOConfig()
    # .api_stack(
    #     enable_rl_module_and_learner=True,
    #     enable_env_runner_and_connector_v2=True,
    # )
    .environment("custom_multi_agent_env", observation_space=obs_space, action_space=action_space)
    .env_runners(
        num_env_runners=1, 
        # remote_worker_envs=True,
        num_envs_per_env_runner=1

    )
    .training(
        lr=1e-3,
        gamma=0.99,
        lambda_=0.95,
        # clip_param=0.2,
        vf_clip_param=1000.0,
        entropy_coeff=0.1,
        train_batch_size=250,  # Adjusted based on expected episode length and number of agents
        sgd_minibatch_size=250,
        num_sgd_iter=1,  # Moderate number of SGD steps
        # _enable_learner_api=False,
    )
    .framework("torch")
    .rollouts(
        # env_runner_cls=MultiAgentEnvRunner,
        num_rollout_workers=5,
        rollout_fragment_length=50,  # Match with avg episode length
        batch_mode="truncate_episodes",
        sample_timeout_s=500  # Allow more time for slow environments
    )
    .resources(num_gpus=1)
    .debugging(log_level="DEBUG")
    .multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        # policies_to_train=None
    )
    # .rl_module(_enable_rl_module_api=False)
)


# Set environment config
config.env_config = {
    "config_path": "config.yaml",
    "ae_folder_path": "home/rppl/Documents/SAAB_thesis/AE_save_06_09"
}

# Build the algorithm
algo = config.build()

for i in range(100000):
    print(f"Iteration: {i}")
    result = algo.train()
    
    # Print training metrics
    print(f"Episodes this iteration: {result.get('episodes_this_iter', 0)}")
    print(f"Timesteps this iteration: {result.get('timesteps_total', 0)}")
    print(f"Episode Reward Mean: {result.get('episode_reward_mean', 0)}")
    print(f"Episode Length Mean: {result.get('episode_len_mean', 0)}")
    # print(f"Agent Steps Sampled: {result.get('num_agent_steps_sampled', 0)}")
    # print(f"Agent Steps Trained: {result.get('num_agent_steps_trained', 0)}")
    # print(f"Env Steps Sampled: {result.get('num_env_steps_sampled', 0)}")
    # print(f"Env Steps Trained: {result.get('num_env_steps_trained', 0)}")
    
    # Save checkpoint every 10 iterations
    if (i + 1) % 10 == 0:
        checkpoint_dir = algo.save(logdir)
        print(f"Checkpoint saved at {checkpoint_dir}")

# Save final checkpoint
final_checkpoint_dir = algo.save(logdir)
print(f"Final checkpoint saved at {final_checkpoint_dir}")

algo.stop()