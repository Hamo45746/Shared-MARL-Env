from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from env import Environment

class MARLlibEnvWrapper(MultiAgentEnv):
    def __init__(self, env):
        self.env = env
        self.num_agents = env.num_agents
        
        # Define action and observation spaces
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        obs = self.env.reset()
        return {i: obs[i] for i in range(self.num_agents)}

    def step(self, action_dict):
        obs, rewards, dones, infos = self.env.step(action_dict)
        
        # Convert to dictionary format expected by MARLlib
        obs = {i: obs[i] for i in range(self.num_agents)}
        rewards = {i: rewards[i] for i in range(self.num_agents)}
        dones = {i: dones for i in range(self.num_agents)}
        dones["__all__"] = all(dones.values())
        infos = {i: infos for i in range(self.num_agents)}
        
        return obs, rewards, dones, infos

    def get_env_info(self):
        return {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": 100,  # Adjust as needed
            "policy_mapping_info": {
                "SearchAndTrack": {
                    "description": "multi-agent search and track",
                    "team_prefix": ("agent_",),
                    "all_agents_one_policy": True,
                    "one_agent_one_policy": True,
                }
            }
        }

# Usage:
# gym_style_env = Environment(config_path)
# marllib_env = MARLlibEnvWrapper(gym_style_env)