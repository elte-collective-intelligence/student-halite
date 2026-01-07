import abc
from src.env.types import State
import numpy as np

class RewardFn(abc.ABC):
    """Abstract class for `Halite` rewards."""

    @abc.abstractmethod
    def __call__(
        self, previous_state: State, new_state: State, action: np.ndarray
    ) -> np.ndarray:
        """The reward function used in the `Halite` environment.

        Args:
            previous_state: Previous state of the environment.
            new_state: New state of the environment.
            action: Array of shape [num_agents, height, width] containing actions taken.
                Actions are: 0=WAIT, 1=UP, 2=RIGHT, 3=DOWN, 4=LEFT
        Returns:
            np.ndarray: Reward for each agent.
        """

