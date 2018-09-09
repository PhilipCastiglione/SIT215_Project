from abc import ABC, abstractmethod

"""
Defines the abstract agent; a simple interface for subclasses
"""
class Agent(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def training_action(self, env, observation):
        pass

    @abstractmethod
    def evaluation_action(self, env, observation):
        pass

    @abstractmethod
    def update(self, observation, action, reward):
        pass
