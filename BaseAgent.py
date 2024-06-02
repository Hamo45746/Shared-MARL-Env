# BaseAgent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):

    def __new__(cls, *args, **kwargs):
        agent = super().__new__(cls)
        return agent

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def __str__(self):
        return f"<{type(self).__name__} instance>"
    
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