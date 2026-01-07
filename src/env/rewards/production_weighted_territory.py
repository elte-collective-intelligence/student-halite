from src.env.rewards.base import RewardFn
from src.env.types import State
import numpy as np

class ProductionWeightedTerritoryRewardFn(RewardFn):
    """Reward function that rewards territory gains weighted by production.
    
    Instead of counting all cells equally, this reward function weights each cell
    by its production value. This encourages agents to prioritize capturing
    high-production cells over low-production cells.
    
    The reward is computed as:
    - For each cell gained: +production_value
    - For each cell lost: -production_value
    
    Rewards are normalized to approximately [-reward_scale, +reward_scale] range
    (default [-5, 5]) for training stability.
    """

    def __init__(self, normalize: bool = True, reward_scale: float = 5.0):
        """Initialize the reward function.
        
        Args:
            normalize: If True, normalize rewards to keep them in a reasonable range.
            reward_scale: Target maximum reward magnitude. Rewards will be scaled to
                         approximately [-reward_scale, +reward_scale] range. Default is 5.0.
        """
        self.normalize = normalize
        self.reward_scale = reward_scale

    def __call__(self, previous_state: State, new_state: State, action: np.ndarray) -> np.ndarray:
        """Reward function based on production-weighted territory changes.
        
        Args:
            previous_state: Previous state of the environment.
            new_state: New state of the environment.
            action: Array of shape [num_agents, height, width] containing actions taken.
            
        Returns:
            np.ndarray: Reward for each agent, weighted by production.
        """
        num_agents = previous_state.alive.shape[0]
        
        ownership_prev = previous_state.grid[0]
        ownership_new = new_state.grid[0]
        production = new_state.grid[2]  # Production values are constant, use from new_state
        
        reward = np.zeros(num_agents, dtype=np.float32)
        
        for agent_id in range(num_agents):
            player_id = agent_id + 1
            
            # Find cells owned by this agent in previous and new states
            prev_owned = (ownership_prev == player_id)
            new_owned = (ownership_new == player_id)
            
            # Cells gained: not owned before, owned now
            cells_gained = (~prev_owned) & new_owned
            
            # Cells lost: owned before, not owned now
            cells_lost = prev_owned & (~new_owned)
            
            # Sum production of gained cells (positive reward)
            production_gained = np.sum(production[cells_gained])
            
            # Sum production of lost cells (negative reward)
            production_lost = np.sum(production[cells_lost])
            
            # Net reward is production gained minus production lost
            reward[agent_id] = production_gained - production_lost
        
        # Normalize to target range [-reward_scale, +reward_scale]
        # This scales rewards so the theoretical maximum (entire grid with max production)
        # maps to reward_scale, keeping rewards in a stable range for training.
        if self.normalize:
            grid_size = production.shape[0] * production.shape[1]
            max_production = np.max(production)
            # Normalize by maximum possible reward, then scale to target range
            max_possible_reward = grid_size * max_production
            if max_possible_reward > 0:
                reward = reward / max_possible_reward * self.reward_scale
        
        return reward

