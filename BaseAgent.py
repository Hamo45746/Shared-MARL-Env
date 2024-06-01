# BaseAgent.py
from abc import ABC, abstractmethod

class BaseAgent(ABC):
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