import gymnasium as gym
from gymnasium import spaces
import numpy as np
from TA_autoencoder import autoencoder

class RLLibEnvWrapper(gym.Env):
    def __init__(self, env, ae_folder_path):
        self.env = env
        self.num_agents = len(self.env.agents)
        self.D = self.env.D  # Number of layers in the state

        # Initialise Autoencoder
        self.autoencoder = autoencoder.EnvironmentAutoencoder()
        self.autoencoder.load_all_autoencoders(ae_folder_path)
        for i in range(3):
            self.autoencoder.autoencoders[i].eval()

        # Define action space
        self.action_space = self.env.action_space

        # Define encoded observation space
        encoded_shape = (256,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.D,) + encoded_shape, 
            dtype=np.float32
        )

    def encode_full_state(self, full_state):
        encoded_full_state = []
        for i in range(self.D):
            if i == 0:
                ae_index = 0  # Use first autoencoder for map layer
            elif i in [1, 2]:
                ae_index = 1  # Use second autoencoder for agent and target layers
            else:
                ae_index = 2  # Use third autoencoder for jammer layer
            
            ae = self.autoencoder.autoencoders[ae_index]
            encoded_full_state.append(ae.encode(full_state[i:i+1]).squeeze())
        
        return np.stack(encoded_full_state)

    def reset(self, seed=None):
        observations = self.env.reset(seed=seed)
        encoded_obs = self._encode_observations(observations)
        return encoded_obs, {}

    def step(self, action_dict):
        observations, rewards, terminated, info = self.env.step(action_dict)
        
        encoded_obs = self._encode_observations(observations)
        
        dones = {"__all__": terminated}
        for agent_id in range(self.num_agents):
            dones[agent_id] = terminated

        return (
            encoded_obs,
            self._format_dict(rewards),
            self._format_dict(dones),
            self._format_dict(info)
        )

    def _encode_observations(self, observations):
        if isinstance(observations, dict):
            return {
                agent_id: self.encode_full_state(obs['full_state'])
                for agent_id, obs in observations.items()
            }
        else:
            return {
                agent_id: self.encode_full_state(obs['full_state'])
                for agent_id, obs in enumerate(observations)
            }

    def _format_dict(self, data):
        if isinstance(data, dict):
            return data
        else:
            return {i: data for i in range(self.num_agents)}

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()