from src.env.rewards.base import RewardFn
from src.env.types import State
import numpy as np

class MinimalReward(RewardFn):
    """Reward function that gives 1 if the game is won, 0 if lost, 0 otherwise.
    
    A game is won by:
    - Eliminating all other players (only one agent remains alive)
    - Reaching time limit and having the maximum territory
    """

    def __call__(self, previous_state: State, new_state: State, action: np.ndarray) -> np.ndarray:
        """Reward function that gives 1 to the winner, 0 to losers, 0 if game hasn't ended.
        
        Args:
            previous_state: Previous state of the environment.
            new_state: New state of the environment.
            action: Array of shape [num_agents, height, width] containing actions taken.
            
        Returns:
            np.ndarray: Reward array with 1 for the winner, 0 for losers, 0 if game continues.
        """
        num_agents = previous_state.alive.shape[0]
        reward = np.zeros(num_agents, dtype=np.float32)  # Initialize all as losers
        
        # Check if game ended by elimination (only one agent alive)
        num_alive_prev = np.sum(previous_state.alive)
        num_alive_new = np.sum(new_state.alive)
        
        if num_alive_new == 1 and num_alive_prev > 1:
            # Game ended by elimination - the surviving agent wins
            winner_idx = np.argmax(new_state.alive)
            reward[winner_idx] = 1.0
            # All others already set to -1.0
            return reward
        
        # Check if game might have ended by time limit
        # We infer this by checking if step_count just reached or passed the time limit
        # Since we don't have direct access to time_limit, we use the same formula as the environment
        grid_size = new_state.grid[0].shape
        estimated_time_limit = int(np.sqrt(grid_size[0] * grid_size[1]) * 10)
        
        # Check if we just reached or passed the time limit (game ended by time limit)
        # This happens when previous step was below limit and current step is at or above limit
        if previous_state.step_count < estimated_time_limit and new_state.step_count >= estimated_time_limit:
            # Game likely ended by time limit - winner is agent with most territory
            ownership_grid = new_state.grid[0]
            territory_counts = np.array([
                np.sum(ownership_grid == (agent_id + 1))
                for agent_id in range(num_agents)
            ], dtype=np.float32)
            
            # Find agent(s) with maximum territory
            max_territory = np.max(territory_counts)
            winners = np.where(territory_counts == max_territory)[0]
            
            # If there's a clear winner (not a tie), give reward
            if len(winners) == 1:
                reward[winners[0]] = 1.0
        
        return reward

