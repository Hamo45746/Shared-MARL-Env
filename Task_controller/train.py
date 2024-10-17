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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
# from ray.rllib.models.tf.tf_action_dist import Deterministic
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.utils.framework import try_import_torch

# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rllib_wrapper import RLLibEnvWrapper

class CustomMetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # Retrieve the first environment instance
        env = base_env.get_sub_environments()[0]
        
        # Call the get_metrics() method from your environment
        metrics = env.get_metrics()

        # Store the returned metrics in episode's custom_metrics dictionary
        for key, value in metrics.items():
            episode.custom_metrics[key] = value


torch, _ = try_import_torch()

class DeterministicWrapper(TorchDistributionWrapper):
    def __init__(self, inputs, model=None):
        super().__init__(inputs, model)
        
        # Ensure that inputs are on the same device as the model
        device = inputs.device if isinstance(inputs, torch.Tensor) else torch.device("cpu")
        
        # Normal distribution with mean = inputs and stddev = 1
        self.dist = torch.distributions.Normal(inputs.to(device), torch.tensor([1.0]).to(device))

    def deterministic_sample(self):
        self.last_sample = self.dist.mean
        return self.last_sample

    def sample(self):
        return self.deterministic_sample()

    def logp(self, actions):
        return self.dist.log_prob(actions).sum(dim=-1)  # Sum over action dimensions

    def entropy(self):
        return self.dist.entropy().sum(dim=-1)  # Sum over action dimensions

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space.shape

ModelCatalog.register_custom_action_dist("deterministic_torch", DeterministicWrapper)


def env_creator(env_config):
    from env import Environment
    env = Environment(config_path=env_config["config_path"])
    return RLLibEnvWrapper(env, ae_folder_path=env_config["ae_folder_path"])

logdir = "./custom_ray_results"
# Update the policies in the config
num_agents = 10  # Adjust based on your environment
obs_shape = (4 * 256,)  # Adjusted for encoded observation space (4 layers + 1 battery layer, 256 encoding size)
# obs_shape = (4, 276, 155)
# action_space = spaces.Box(low=np.array([-20, -20]), high=np.array([20, 20]), dtype=np.int32)
action_space = spaces.Discrete((2 * 20 + 1) ** 2)

obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 * 256,), dtype=np.float32)
# obs_space =  spaces.Box(low=-20.0, high=1.0, shape=(4, 276, 155), dtype=np.float32)

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
        # exploration_config= {"type": "EpsilonGreedy", "initial_epsilon": 1.0, "final_epsilon": 0.02, "epsilon_timesteps": 50000},
        gamma=0.99,
        lambda_=0.95,
        # clip_param=100,
        vf_clip_param=1000,
        # entropy_coeff_schedule=[(0, 0.9), (100000, 0.5), (1000000,0.2), (10000000, 0.01)],
        entropy_coeff=0.1,
        train_batch_size=50,  # Adjusted based on expected episode length and number of agents
        sgd_minibatch_size=50, # These were 625 when using multiple workers
        num_sgd_iter=1,  # Moderate number of SGD steps
        # _enable_learner_api=False,
        # model = {
        #     "conv_filters": [
        #         [16, [4, 4], 2],
        #         [32, [4, 4], 2],
        #         [64, [4, 4], 2],
        #         [128, [4, 4], 2],
        #         [256, [4, 4], 2],
        #         [512, [1, 9], 1]
        #     ]
        # }   
        model = {
            "fcnet_hiddens": [1024, 1024],
            # "custom_action_dist": "deterministic_torch"
        }
    )
    .framework("torch")
    # .rollouts(
    #     # env_runner_cls=MultiAgentEnvRunner,
    #     num_rollout_workers=5,
    #     rollout_fragment_length=125,  # Match with avg episode length
    #     batch_mode="truncate_episodes",
    #     sample_timeout_s=500  # Allow more time for slow environments
    # )
    .rollouts(
        # env_runner_cls=MultiAgentEnvRunner,
        num_rollout_workers=0,
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
    .callbacks(CustomMetricsCallback)
    # .rl_module(_enable_rl_module_api=False)
)


# Set environment config
config.env_config = {
    "config_path": "config.yaml",
    "ae_folder_path": "home/rppl/Documents/SAAB_thesis/AE_save_06_09"
}

# Build the algorithm
algo = config.build()

for i in range(1000000):
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