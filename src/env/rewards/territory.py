from src.env.rewards.base import RewardFn
from src.env.types import State
import numpy as np

class TerritoryRewardFn(RewardFn):
    """Reward function that rewards the agent for controlling more territory and penalizes for losing territory."""

    def __call__(self, previous_state: State, new_state: State, action: np.ndarray) -> np.ndarray:
        """Reward function based on the territory controlled by each agent now minus the one controlled before."""

        num_agents = previous_state.alive.shape[0]
        
        prev_owned = np.arange(1, num_agents + 1)[:, None, None] == previous_state.grid[0]
        new_owned = np.arange(1, num_agents + 1)[:, None, None] == new_state.grid[0]
        
        prev_counts = np.sum(prev_owned, axis=(1, 2), dtype=np.float32)
        new_counts = np.sum(new_owned, axis=(1, 2), dtype=np.float32)
        
        return new_counts - prev_counts

