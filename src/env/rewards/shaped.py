from src.env.rewards.base import RewardFn
from src.env.rewards.territory import TerritoryRewardFn
from src.env.rewards.strength import StrengthRewardFn
from src.env.rewards.production import ProductionRewardFn
from src.env.rewards.minimal import MinimalReward
from src.env.types import State
import numpy as np

class ShapedRewardFn(RewardFn):
    """A well-designed reward function that combines multiple signals for better learning.
    
    This reward function:
    - Combines territory, strength, and production rewards with appropriate weights
    - Normalizes rewards to keep them in a reasonable range for stable learning
    - Adds a win bonus to encourage winning
    - Rewards expansion into high-production areas
    - Penalizes territory losses more heavily than gains
    
    This is designed to provide dense, well-scaled rewards that help agents learn
    effective strategies while maintaining stable gradients.
    """
    
    def __init__(
        self,
        territory_weight: float = 1.0,
        strength_weight: float = 0.05,
        production_weight: float = 0.3,
        expansion_bonus: float = 0.5,
        loss_penalty_multiplier: float = 1.5,
        zero_strength_move_penalty: float = 0.1,
        normalize: bool = True,
    ):
        """Initialize the shaped reward function.
        
        Args:
            territory_weight: Weight for territory rewards (most important)
            strength_weight: Weight for strength rewards (lower, as strength can be noisy)
            production_weight: Weight for production rewards
            expansion_bonus: Bonus multiplier for expanding into high-production cells
            loss_penalty_multiplier: Penalty multiplier for losing territory (asymmetric)
            zero_strength_move_penalty: Penalty for moving cells with strength = 0 (wasteful moves)
            normalize: Whether to normalize rewards by grid size
            reward_scale: Additional scaling factor to keep rewards in a smaller range (default 0.1)
        """
        self.territory_weight = territory_weight
        self.strength_weight = strength_weight
        self.production_weight = production_weight
        self.expansion_bonus = expansion_bonus
        self.loss_penalty_multiplier = loss_penalty_multiplier
        self.zero_strength_move_penalty = zero_strength_move_penalty
        self.normalize = normalize
        
        # Initialize component reward functions
        self.territory_fn = TerritoryRewardFn()
        self.strength_fn = StrengthRewardFn()
        self.production_fn = ProductionRewardFn()

    
    def __call__(self, previous_state: State, new_state: State, action: np.ndarray) -> np.ndarray:
        """Compute the shaped reward for each agent.
        
        Args:
            previous_state: Previous state of the environment.
            new_state: New state of the environment.
            action: Array of shape [num_agents, height, width] containing actions taken.
            
        Returns:
            np.ndarray: Reward for each agent.
        """
        num_agents = previous_state.alive.shape[0]
        grid_size = previous_state.grid[0].shape[0] * previous_state.grid[0].shape[1]
        
        # Compute base rewards
        territory_reward = self.territory_fn(previous_state, new_state, action)
        strength_reward = self.strength_fn(previous_state, new_state, action)
        production_reward = self.production_fn(previous_state, new_state, action)
        
        # Normalize if requested
        if self.normalize:
            territory_reward = territory_reward / grid_size
            strength_reward = strength_reward / (255.0 * grid_size)  # Max strength is 255
            production_reward = production_reward / grid_size
        
        # Apply asymmetric loss penalty (losing territory hurts more)
        territory_gains = np.maximum(territory_reward, 0)
        territory_losses = np.minimum(territory_reward, 0)
        territory_reward = territory_gains + territory_losses * self.loss_penalty_multiplier
        
        # Add expansion bonus for capturing high-production cells
        expansion_bonus = self._compute_expansion_bonus(previous_state, new_state)
        
        # Penalize moving cells with strength = 0
        zero_strength_penalty = self._compute_zero_strength_move_penalty(previous_state, new_state, action)
        
        # Combine rewards with weights
        reward = (
            self.territory_weight * territory_reward +
            self.strength_weight * strength_reward +
            self.production_weight * production_reward +
            expansion_bonus -
            zero_strength_penalty
        )
        
        
        return reward.astype(np.float32)
    
    def _compute_expansion_bonus(self, previous_state: State, new_state: State) -> np.ndarray:
        """Compute bonus for expanding into high-production cells.
        
        This rewards agents for capturing cells with high production values,
        which encourages strategic expansion.
        """
        num_agents = previous_state.alive.shape[0]
        grid_size = previous_state.grid[0].shape[0] * previous_state.grid[0].shape[1]
        bonus = np.zeros(num_agents, dtype=np.float32)
        
        ownership_prev = previous_state.grid[0]
        ownership_new = new_state.grid[0]
        production = new_state.grid[2]
        
        for agent_id in range(num_agents):
            player_id = agent_id + 1
            
            # Find cells that were just captured (not owned before, owned now)
            just_captured = (ownership_prev != player_id) & (ownership_new == player_id)
            
            # Sum production of newly captured cells
            captured_production = np.sum(production[just_captured])
            
            # Normalize and apply bonus
            if self.normalize:
                bonus[agent_id] = self.expansion_bonus * captured_production / grid_size
            else:
                bonus[agent_id] = self.expansion_bonus * captured_production * 0.01  # Small scale
        
        return bonus
    
    def _compute_zero_strength_move_penalty(self, previous_state: State, new_state: State, action: np.ndarray) -> np.ndarray:
        """Compute penalty for moving cells with strength = 0.
        
        This penalizes agents for moving cells that have no strength, which is
        wasteful since cells with 0 strength cannot contribute effectively to moves.
        Uses the actual actions taken to detect when zero-strength cells were moved.
        """
        num_agents = previous_state.alive.shape[0]
        grid_size = previous_state.grid[0].shape[0] * previous_state.grid[0].shape[1]
        penalty = np.zeros(num_agents, dtype=np.float32)
        
        ownership_prev = previous_state.grid[0]
        strength_prev = previous_state.grid[1]
        
        for agent_id in range(num_agents):
            player_id = agent_id + 1
            
            # Find cells that were owned by this agent with strength = 0
            zero_strength_owned = (ownership_prev == player_id) & (strength_prev == 0)
            
            # Get actions for this agent (action != 0 means moving, 0 means staying)
            agent_actions = action[agent_id]
            moving_cells = (agent_actions != 0)
            
            # Count how many zero-strength cells were moved (wasteful)
            zero_strength_moved = zero_strength_owned & moving_cells
            num_moved = np.sum(zero_strength_moved)
            
            # Apply penalty (normalized if requested)
            if self.normalize:
                penalty[agent_id] = self.zero_strength_move_penalty * num_moved / grid_size
            else:
                penalty[agent_id] = self.zero_strength_move_penalty * num_moved * 0.01  # Small scale
        
        return penalty


