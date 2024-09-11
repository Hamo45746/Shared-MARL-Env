import torch
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from TA_autoencoder import autoencoder
import gymnasium as gym

class RLLibEnvWrapper(MultiAgentEnv):
    def __init__(self, env, ae_folder_path):
        self.env = env
        self.num_agents = len(self.env.agents)
        self.D = self.env.D  # Number of layers in the state
        self.step_count = 0

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize Autoencoder
        self.autoencoder = autoencoder.EnvironmentAutoencoder()
        self.autoencoder.load_all_autoencoders(ae_folder_path)
        for i in range(3):
            self.autoencoder.autoencoders[i].to(self.device)
            self.autoencoder.autoencoders[i].eval()

        # Define action and observation spaces
        self.action_space = self.env.action_space
        encoded_shape = (256,)
        self.observation_space = gym.spaces.Box(
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
            # Convert to float32, add batch dimension, and move to device
            input_tensor = torch.from_numpy(full_state[i:i+1]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                encoded_layer = ae.encode(input_tensor).cpu().squeeze().numpy()
            encoded_full_state.append(encoded_layer)
        
        # Add battery information as a repeated 256-element vector
        battery_vector = np.full(256, battery, dtype=np.float32)
        encoded_full_state.append(battery_vector)
        
        return np.stack(encoded_full_state)

    def reset(self, *, seed=None, options=None):
        print("RLLibEnvWrapper reset called")
        observations, info = self.env.reset(seed=seed, options=options)
        battery_levels = self.env.get_battery_levels()
        encoded_obs = self._encode_observations(observations, battery_levels)
        self.step_count = 0
        print(f"Reset returned observations for {len(encoded_obs)} agents")
        return encoded_obs, info

    def step(self, action_dict):
        print(f"RLLibEnvWrapper step called with actions for {len(action_dict)} agents")
        observations, rewards, terminated, truncated, info = self.env.step(action_dict)
        battery_levels = self.env.get_battery_levels()
        encoded_obs = self._encode_observations(observations, battery_levels)

        self.step_count += 1
        
        print(f"Step returned: obs={len(encoded_obs)}, rewards={len(rewards)}, terminated={terminated['__all__']}")
        return encoded_obs, rewards, terminated, truncated, info

    def _encode_observations(self, observations, battery_levels):
        encoded_observations = {}
        for agent_id, obs in observations.items():
            if self.env.agents[agent_id].is_terminated():
                encoded_observations[agent_id] = np.zeros((self.D + 1, 256), dtype=np.float32)
            else:
                full_state = obs['full_state']
                battery = battery_levels[agent_id]
                encoded_observations[agent_id] = self.encode_full_state(full_state, battery)
        return encoded_observations

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()