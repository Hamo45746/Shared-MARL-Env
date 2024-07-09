# This file / class is currently unused. THe intended purpose was to have it to pass rewards,
# however currently the reward.py file is importing directly to env. It was also meant to be used to select actions, but currently that 
# is happening in continuous_agent.py

import numpy as np
from continuous_agent import ContinuousAgent
from agent_utils import create_agents
from Continuous_controller import reward

class AgentController:
    def __init__(self, config):
        self.config = config
        self.randomiser = np.random.RandomState(self.config['seed'])
        self.agents = []
        self.init_agents()

    def init_agents(self):
        agent_positions = self.config.get('agent_positions', None)
        self.agents = create_agents(self.config['n_agents'], self.config['map_matrix'], self.config['obs_range'], self.randomiser, agent_positions, randinit=True)

    # def get_agent_actions(self):
    #     actions = []
    #     for agent in self.agents:
    #         action = agent.get_next_action()
    #         actions.append(action)
    #     return actions

    # def update_agent_positions(self, actions):
    #     for agent, action in zip(self.agents, actions):
    #         agent.step(action)

    # def calculate_reward(agent, env):
    #     agent_reward = reward.calculate_continuous_reward(agent, env)
    #     return agent_reward

    
            
        
    
