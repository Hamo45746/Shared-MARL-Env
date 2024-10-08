import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.annotations import override
import numpy as np

class CentralisedCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Define the critic input dimensions
        self.num_agents = 5  # Example with 5 agents
        self.obs_dim = model_config["custom_model_config"]["obs_dim"]  # Assumed to be (4,32) per agent
        self.act_dim =  model_config["custom_model_config"]["act_dim"]  # Action space of each agent

        # Input dimension for centralized critic (obs + actions for all agents)
        self.critic_input_dim = 654

        # Centralized critic network
        self.central_critic_net = nn.Sequential(
            nn.Linear(self.critic_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output value prediction
        )

        # Policy network for each individual agent
        self.policy_net = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name="policy_net")

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']

        # Extract parts of the observation
        encoded_map = obs['encoded_map']  # Shape: (batch_size, 4, 32)
        velocity = obs['velocity']        # Shape: (batch_size, 2)
        goal = obs['goal']                # Shape: (batch_size, 2)
        other_agents_obs = obs['other_agents_obs']  # Shape: (batch_size, 5, 4, 32)

        # Flatten the encoded map
        encoded_map_flat = encoded_map.view(encoded_map.size(0), -1)
        other_agents_obs_flat = other_agents_obs.view(other_agents_obs.size(0), -1)  # Flatten other agents' observations
        
        # Concatenate the observations and actions for all agents
        all_obs_concat = torch.cat([encoded_map_flat, velocity, goal, other_agents_obs_flat], dim=-1) 

        # Store the concatenated observations for the value function
        self._obs_concat = all_obs_concat
        
        return self.policy_net({'obs': all_obs_concat}, state, seq_lens)

    def central_value_function(self, obs, actions):
        # Assuming obs shape is already concatenated and passed as (batch_size, all agents' obs)
        critic_input = torch.cat([obs, actions], dim=-1)
        return self.central_critic_net(critic_input)

    @override(TorchModelV2)
    def value_function(self):
        actions = getattr(self, '_actions_concat', torch.zeros((self._obs_concat.size(0), self.act_dim * self.num_agents), device=self._obs_concat.device))
        # Print the shape of the concatenated observations and actions
        return self.central_value_function(self._obs_concat, actions)
