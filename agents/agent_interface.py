"""Agent interface for frequency control environment."""
from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    """Abstract agent class."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Compute an action given an observation."""
        raise NotImplementedError

    @abstractmethod
    def learn(self, *args, **kwargs) -> None:
        """Update the agent based on experience."""
        raise NotImplementedError


class RandomAgent(Agent):
    """Simple agent that samples random actions from the environment."""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation: Any) -> Any:  # pylint: disable=unused-argument
        return self.action_space.sample()

    def learn(self, *args, **kwargs) -> None:  # pylint: disable=unused-argument
        pass
