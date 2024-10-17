import torch
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from TA_autoencoder import autoencoder
import gymnasium as gym

class RLLibEnvWrapper(MultiAgentEnv):
    def __init__(self, env, ae_folder_path):
        super().__init__()
        self.env = env
        self.num_agents = len(self.env.agents)
        self.D = self.env.D  # Number of layers in the state
        self._agent_ids = set(range(self.num_agents))

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialise Autoencoder
        self.autoencoder = autoencoder.EnvironmentAutoencoder()
        self.autoencoder.load_all_autoencoders(ae_folder_path)
        for i in range(3):
            self.autoencoder.autoencoders[i].to(self.device)
            self.autoencoder.autoencoders[i].eval()

        # Define action and observation spaces
        self._observation_spaces = {
            i: gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4 * 256,), dtype=np.float32)
            for i in range(self.num_agents)
        }

        self._action_spaces = {
            i: gym.spaces.Discrete((2 * 20 + 1) ** 2)
            for i in range(self.num_agents)
        }
        # self._action_spaces = {
        #     i: gym.spaces.Discrete((2 * 20 + 1) ** 2)
        #     for i in range(self.num_agents)
        # }

        # Define combined spaces for MultiAgentEnv
        self.observation_space = gym.spaces.Dict(self._observation_spaces)
        self.action_space = gym.spaces.Dict(self._action_spaces)
        
    # @property
    # def action_space(self):
    #     return self._action_spaces

    # @property
    # def observation_space(self):
    #     return self._observation_spaces

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
        # battery_vector = np.full(256, battery, dtype=np.float32)
        # encoded_full_state.append(battery_vector)
        
        return np.concatenate(encoded_full_state).flatten()

    def reset(self, *, seed=None, options=None):
        print("RLLibEnvWrapper reset called")
        observations, _ = self.env.reset(seed=seed, options=options)
        battery_levels = self.env.get_battery_levels()
        encoded_obs = self._encode_observations(observations, battery_levels)
        print(f"Reset returned observations for {len(observations)} agents")
        return encoded_obs, {}

    def step(self, action_dict):
        # print(f"RLLibEnvWrapper step called with actions for {len(action_dict)} agents")
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        battery_levels = self.env.get_battery_levels()
        encoded_obs = self._encode_observations(obs, battery_levels)

        if terminated["__all__"]:
            for agent_id in action_dict:
                info[agent_id] = {"episode_done": True}
            
        print(f"Step returned: obs={len(obs)}, rewards={len(rewards)}, terminated={terminated['__all__']}")
        return encoded_obs, rewards, terminated, truncated, info

    def _encode_observations(self, observations, battery_levels):
        encoded_observations = {}
        for agent_id, obs in observations.items():
            if self.env.agents[agent_id].is_terminated():
                # Terminated state: map layer = all 1's, other layers = -20's, battery = 0
                terminated_state = np.zeros((4, 256), dtype=np.float32)
                terminated_state[0, :] = 1.0  # Map layer with all 1's
                terminated_state[1:4, :] = -20.0  # Other layers with -20's
                # terminated_state[4, :] = 0.0  # Battery layer with 0
                encoded_observations[agent_id] = terminated_state.flatten()
            else:
                full_state = obs['full_state']
                battery = battery_levels[agent_id]
                encoded_observations[agent_id] = self.encode_full_state(full_state, battery)
        return encoded_observations

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
    
    def get_metrics(self):
        # Call the environment's get_metrics function.
        return self.env.get_metrics()