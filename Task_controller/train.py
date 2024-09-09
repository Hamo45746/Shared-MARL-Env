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


    def encode_full_state(self, full_state, battery):
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
        
        # Add battery information as a repeated 256-element vector
        battery_vector = np.full(256, battery, dtype=np.float32)
        encoded_full_state.append(battery_vector)
        
        return np.stack(encoded_full_state)
    

    def reset(self, seed=None):
        observations = self.env.reset(seed=seed)
        battery_levels = self.env.get_battery_levels()
        encoded_obs = self._encode_observations(observations, battery_levels)
        return encoded_obs, {}


    def step(self, action_dict):
        observations, rewards, episode_done, info = self.env.step(action_dict)
        battery_levels = self.env.get_battery_levels()
        encoded_obs = self._encode_observations(observations, battery_levels)
        
        dones = {}
        for agent_id in range(self.num_agents):
            dones[agent_id] = self.env.agents[agent_id].is_terminated()
        
        # Set __all__ to True only if all agents are terminated
        dones["__all__"] = episode_done

        return encoded_obs, self._format_dict(rewards), dones, self._format_dict(info)


    def _encode_observations(self, observations, battery_levels):
        encoded_observations = {}

        for agent_id, obs in observations.items():
            if self.env.agents[agent_id].is_terminated():
                # For terminated agents, return a zero-filled observation
                encoded_observations[agent_id] = np.zeros((self.D + 1, 256), dtype=np.float32)
            else:
                # For active agents, encode the full state and include battery level
                full_state = obs['full_state']
                battery = battery_levels[agent_id]
                encoded_observations[agent_id] = self.encode_full_state(full_state, battery)

        return encoded_observations
    

    def _format_dict(self, data):
        if isinstance(data, dict):
            return data
        else:
            return {i: data for i in range(self.num_agents)}

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()