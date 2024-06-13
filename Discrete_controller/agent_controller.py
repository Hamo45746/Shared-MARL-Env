import numpy as np
from discrete_agent import DiscreteAgent
from continuous_agent import ContinuousAgent
from agent_utils import create_agents


class DiscreteAgentController:
   def __init__(self, config, agent_type='discrete'):
        self.config = config
        self.randomiser = np.random.RandomState(self.config['seed'])
        self.agent_type = agent_type
        self.agents = []
        self.init_agents()

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

   # Placeholder reward
   def calculate_reward(agent):
      reward = 1
      return reward 
   