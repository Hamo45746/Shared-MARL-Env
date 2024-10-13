import sys
import os
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env")))
from rllib_env import Environment
import yaml
import pickle
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from centralised_critic_model import CentralisedCriticModel

# Register the custom centralized critic model
ModelCatalog.register_custom_model("centralised_critic_model", CentralisedCriticModel)
config_path = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/config.yaml' 
checkpoint_dir = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/custom_ray_results/'
#checkpoint_dir = '/Users/alexandramartinwallace/Documents/Uni/METR4911/Working/Shared-MARL-Env/outputs/10of10/custom_ray_results/'
params_path = '/Users/alexandramartinwallace/ray_results/PPO_custom_multi_agent_env_2024-10-12_19-26-054568b3f6/params.pkl'
env2 = Environment(config_path)
# Make sure to load the environment registration
register_env("custom_multi_agent_env", lambda config: Environment(config_path=config["config_path"], render_mode=config.get("render_mode", "human")))

# Run the simulation with the policy
env2.run_simulation_with_policy(checkpoint_dir=checkpoint_dir, params_path=params_path, max_steps=900)
