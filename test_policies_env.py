import os
import sys
from Task_controller.rllib_wrapper import RLLibEnvWrapper
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import numpy as np
import pickle
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module import RLModule
from gymnasium import spaces

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



config_path = '/home/rppl/Documents/SAAB_thesis/Shared-MARL-Env/config.yaml'
checkpoint_dir = '/home/rppl/Documents/SAAB_thesis/Shared-MARL-Env/custom_ray_results/' # TODO: CHANGE THESE PATHS 
params_path = '/home/rppl/ray_results/PPO_custom_multi_agent_env_2024-09-26_19-36-27leycfxt5/params.pkl'

num_agents = 10


def env_creator(env_config):
    from env import Environment
    env = Environment(config_path=env_config["config_path"])
    return RLLibEnvWrapper(env, ae_folder_path=env_config["ae_folder_path"])

def policy_mapping_fn(agent_id, **kwargs):
    # This function maps each agent to a policy based on its agent_id
    return f"policy_{agent_id}"


# Create the environment
env_config = {
    "config_path": config_path,
    "ae_folder_path": "/media/rppl/T7 Shield/METR4911/TA_autoencoder_h5_data/AE_save_06_09"
}
env = env_creator(env_config)



def run_simulation_with_policy(env, checkpoint_dir, params_path, max_steps=100, iteration=None):
    # Load the configuration from the params.pkl file
    with open(params_path, "rb") as f:
        config = pickle.load(f)

    # Register the custom environment
    register_env("custom_multi_agent_env", env_creator)

    # Recreate the trainer with the loaded configuration
    # trainer = config.build()
    algo = PPO(config=config)
    # Restore the checkpoint from the checkpoint directory
    # trainer.restore(checkpoint_dir)
    algo.restore(checkpoint_dir)
    # trainer = Algorithm.from_checkpoint(checkpoint_dir)

    obs_dict, _ = env.reset()
    running = True
    step_count = 0
    # collected_data = []

    while running and step_count < max_steps:
        action_dict = {}
        for agent_id, obs in obs_dict.items():
            # action = trainer.compute_single_action(obs, policy_id=policy_mapping_fn(agent_id), explore=True)
            # action = algo.compute_single_action(obs, policy_id=policy_mapping_fn(agent_id), explore=True)
            policy = algo.get_policy(f"policy_{agent_id}")
            action = policy.compute_single_action(obs, explore=True, clip_action=True)[0]
            action_dict[agent_id] = action
        # action = trainer.compute_actions(obs_dict, explore=False)
 
        obs_dict, rewards, terminated, truncated, info = env.step(action_dict)
        # collected_data.append({
        #     'observations': obs,
        #     'rewards': rewards,
        #     'terminated': terminated,
        #     'truncated': truncated,
        #     'info': info
        # })
        step_count += 1

        if terminated.get("__all__", False) or truncated.get("__all__", False):
            break
        
    # if iteration is not None:
    #     map_filename = f"outputs/environment_snapshot_iteration_{iteration}.png"
    # else:
    #     map_filename = "outputs/environment_snapshot.png"

    # return collected_data

# Run the simulation
run_simulation_with_policy(env, checkpoint_dir=checkpoint_dir, params_path=params_path, max_steps=50)
