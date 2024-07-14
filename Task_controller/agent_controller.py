import numpy as np
from discrete_agent import DiscreteAgent
from continuous_agent import ContinuousAgent
from agent_utils import create_agents

## DONT THINK THIS FILE IS NEEDED ##
class DiscreteAgentController:
   def __init__(self, config, agent_type='discrete'):
      self.config = config
      self.randomiser = np.random.RandomState(self.config['seed'])
      self.agent_type = agent_type
      self.agents = []
      #   self.init_agents()

   def init_agents(self):
      agent_positions = self.config.get('agent_positions', None)
      if self.agent_type == 'discrete':
         self.agents = create_agents(
               self.config['n_agents'],
               self.config['map_matrix'],
               self.config['obs_range'],
               self.randomiser,
               agent_positions,
               randinit=True,
               agent_class=DiscreteAgent
            )
      elif self.agent_type == 'continuous':
            self.agents = create_agents(
               self.config['n_agents'],
               self.config['map_matrix'],
               self.config['obs_range'],
               self.randomiser,
               agent_positions,
               randinit=True,
               agent_class=ContinuousAgent
            )

   def get_agent_actions(self):
         actions = []
         for agent in self.agents:
            action = agent.get_next_action()
            actions.append(action)
         return actions

   def update_agent_positions(self, actions):
      for agent, action in zip(self.agents, actions):
         agent.step(action)

   def calculate_reward(agent):
      return 1
   # @staticmethod
   # def calculate_reward(agent, env):
   # # Get the agent's current position and battery level
   #    pos = agent.current_position()
   #    battery_level = env.agent_layer.get_battery_level(env.agent_name_mapping[agent])

   #    # Initialise reward
   #    reward = 0

   #    # Reward for exploring new areas
   #    if env.agent_layer.layer_state[pos[0], pos[1]] == battery_level:  # If this is a new position
   #          reward += 0.1

   #    # Reward for being close to a target
   #    closest_target_dist = float('inf')
   #    for target in env.target_layer.targets:
   #       dist = np.linalg.norm(np.array(pos) - np.array(target.current_position()))
   #       closest_target_dist = min(closest_target_dist, dist)
        
   #    if closest_target_dist <= env.obs_range:
   #       reward += 1 / (closest_target_dist + 1)  # Inverse reward based on distance

   #    # Penalty for low battery
   #    #   if battery_level < 20:
   #    #       reward -= 0.5
   #    #   elif battery_level < 50:
   #    #       reward -= 0.2

   #      # Reward for destroying jammers
   #    for jammer in env.jammer_layer.jammers:
   #       if jammer.get_destroyed() and np.array_equal(pos, jammer.current_position()):
   #          reward += 5

   #    # Adjust reward based on battery level
   #    # reward *= (battery_level / 100)  # Scale reward by battery percentage

   #    return reward

   def is_episode_done(self, env):
      return all(env.agent_layer.is_agent_terminated(i) for i in range(len(self.agents)))