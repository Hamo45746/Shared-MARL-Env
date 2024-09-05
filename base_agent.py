# BaseAgent.py
from abc import ABC, abstractmethod
from gymnasium import spaces
# from gym import spaces # for MARLlib

class BaseAgent(ABC):

    def __new__(cls, *args, **kwargs):
        agent = super().__new__(cls)
        return agent

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError()
    
    @abstractmethod
    def get_next_action(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def set_position(self, x, y):
        pass

    @abstractmethod
    def current_position(self):
        pass

    @abstractmethod
    def update_local_state(self, observed_state, observer_position):
        """Update the agent's local state based on observed data."""
        pass

    @abstractmethod
    def set_observation_state(self, observation):
        """Set the agent's observation state."""
        pass

    def __str__(self):
        return f"<{type(self).__name__} instance>"