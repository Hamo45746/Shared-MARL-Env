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
        self.num_agents = model_config["custom_model_config"]["num_agents"]  # Example with 5 agents
        self.obs_dim = model_config["custom_model_config"]["obs_dim"]  # Assumed to be (4,32) per agent
        self.act_dim =  model_config["custom_model_config"]["act_dim"]  # Action space of each agent

        # Policy network for each individual agent
        self.policy_net = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name="policy_net")
        
        # Define the network architecture for centralized value function
        #self.critic_input_dim = self.obs_dim * self.num_agents + self.act_dim * self.num_agents 
        self.critic_input_dim = 274

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


    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']

        # Extract parts of the observation
        encoded_map = obs['encoded_map'].view(obs['encoded_map'].size(0), -1)  # Shape: (batch_size, 4, 32)
        velocity = obs['velocity']        # Shape: (batch_size, 2)
        goal = obs['goal']                # Shape: (batch_size, 2)

        # Concatenate the observations and actions for all agents
        all_obs_concat = torch.cat([encoded_map, velocity, goal], dim=-1) 

        # Store the concatenated observations for the value function
        self._obs_concat = all_obs_concat
        
        return self.policy_net({'obs': all_obs_concat}, state, seq_lens)

    def central_value_function(self, obs, global_obs, global_actions):
        # Concatenate local obs with global obs and global actions
        critic_input = torch.cat([obs, global_obs, global_actions], dim=-1)
        return self.central_critic_net(critic_input)

    @override(TorchModelV2)
    def value_function(self):
        # global_obs and global_actions should be provided from the postprocessing function
        global_obs = getattr(self, '_global_obs', torch.zeros_like(self._obs_concat))
        global_actions = getattr(self, '_global_actions', torch.zeros((self._obs_concat.size(0), self.act_dim * self.num_agents), device=self._obs_concat.device))

        return self.central_value_function(self._obs_concat, global_obs, global_actions)