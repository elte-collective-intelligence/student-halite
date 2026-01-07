from src.env.rewards.base import RewardFn
from src.env.types import State
import numpy as np

class ProductionRewardFn(RewardFn):
    """Reward function that rewards the agent for increasing production in controlled territory."""

    def __call__(self, previous_state: State, new_state: State, action: np.ndarray) -> np.ndarray:
        """Reward function based on the production difference in controlled territory."""
        num_agents = previous_state.alive.shape[0]
        
        prev_owned = np.arange(1, num_agents + 1)[:, None, None] == previous_state.grid[0]
        new_owned = np.arange(1, num_agents + 1)[:, None, None] == new_state.grid[0]
        
        prev_production = previous_state.grid[2]
        new_production = new_state.grid[2]
        
        prev_production_sum = np.sum(
            prev_owned * prev_production, 
            axis=(1, 2), 
            dtype=np.float32
        )
        new_production_sum = np.sum(
            new_owned * new_production, 
            axis=(1, 2), 
            dtype=np.float32
        )
        
        return new_production_sum - prev_production_sum

