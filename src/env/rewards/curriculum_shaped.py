from src.env.rewards.base import RewardFn
from src.env.rewards.shaped import ShapedRewardFn
from src.env.rewards.minimal import MinimalReward
from src.env.types import State
import numpy as np
from typing import Optional


class CurriculumShapedRewardFn(RewardFn):
    """Curriculum-shaped reward that blends shaped and minimal rewards over time.
    
    This reward function implements scheduled/annealed reward shaping:
    - Early training: Heavy territory and survival shaping (w(t) ≈ 1.0)
    - Mid training: Reduce regularizers, downweight per-step rewards
    - Late training: Mostly win/loss or sparse outcome-based rewards (w(t) ≈ 0.0)
    
    The reward is computed as:
        R_t = w(t) · R_shaped + (1 - w(t)) · R_minimal
    
    where w(t) decreases over time according to a schedule.
    
    This combines fast early learning with correct final incentives, reduces
    reward hacking, and helps transfer and generalization.
    """
    
    def __init__(
        self,
        shaped_reward_fn: Optional[RewardFn] = None,
        minimal_reward_fn: Optional[RewardFn] = None,
        schedule_type: str = "linear",
        start_episode: int = 0,
        end_episode: int = 10000,
        start_weight: float = 1.0,
        end_weight: float = 0.0,
        schedule_kwargs: Optional[dict] = None,
    ):
        """Initialize the curriculum-shaped reward function.
        
        Args:
            shaped_reward_fn: The shaped reward function to use (default: ShapedRewardFn).
            minimal_reward_fn: The minimal reward function to use (default: MinimalReward).
            schedule_type: Type of schedule for w(t). Options: "linear", "exponential", "cosine", "step".
            start_episode: Episode number when curriculum starts (w(t) = start_weight).
            end_episode: Episode number when curriculum ends (w(t) = end_weight).
            start_weight: Initial weight for shaped reward (default: 1.0).
            end_weight: Final weight for shaped reward (default: 0.0).
            schedule_kwargs: Additional keyword arguments for the schedule function.
        """
        self.shaped_reward_fn = shaped_reward_fn or ShapedRewardFn()
        self.minimal_reward_fn = minimal_reward_fn or MinimalReward()
        
        self.schedule_type = schedule_type
        self.start_episode = start_episode
        self.end_episode = end_episode
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.schedule_kwargs = schedule_kwargs or {}
        
        # Current episode (updated externally via update_episode)
        self.current_episode = 0
        
        # Validate schedule type
        valid_schedules = ["linear", "exponential", "cosine", "step"]
        if schedule_type not in valid_schedules:
            raise ValueError(f"schedule_type must be one of {valid_schedules}, got {schedule_type}")
    
    def update_episode(self, episode: int):
        """Update the current episode number for curriculum scheduling.
        
        This should be called at the start of each training episode to update
        the curriculum weight based on the current training progress.
        
        Args:
            episode: Current episode number (0-indexed).
        """
        self.current_episode = episode
    
    def _compute_weight(self, episode: int) -> float:
        """Compute the curriculum weight w(t) for the given episode.
        
        Args:
            episode: Current episode number.
            
        Returns:
            Weight w(t) in [0, 1] for blending shaped and minimal rewards.
        """
        # Before start: use start_weight
        if episode < self.start_episode:
            return self.start_weight
        
        # After end: use end_weight
        if episode >= self.end_episode:
            return self.end_weight
        
        # During curriculum: compute based on schedule
        progress = (episode - self.start_episode) / (self.end_episode - self.start_episode)
        progress = np.clip(progress, 0.0, 1.0)
        
        if self.schedule_type == "linear":
            weight = self.start_weight + (self.end_weight - self.start_weight) * progress
        
        elif self.schedule_type == "exponential":
            # Exponential decay
            decay_rate = self.schedule_kwargs.get("decay_rate", 1.0)
            if self.start_weight == 0:
                weight = 0.0
            elif self.end_weight == 0:
                # Exponential decay to zero: w(t) = start * exp(-decay_rate * progress)
                weight = self.start_weight * np.exp(-decay_rate * progress)
            else:
                # General exponential: w(t) = start * (end/start)^progress
                weight = self.start_weight * (self.end_weight / self.start_weight) ** (progress * decay_rate)
        
        elif self.schedule_type == "cosine":
            # Cosine annealing: smooth transition
            weight = self.end_weight + (self.start_weight - self.end_weight) * 0.5 * (1 + np.cos(np.pi * progress))
        
        elif self.schedule_type == "step":
            # Step function: discrete transitions
            num_steps = self.schedule_kwargs.get("num_steps", 5)
            step_size = 1.0 / num_steps
            step = int(progress / step_size)
            step = min(step, num_steps - 1)
            step_progress = (progress - step * step_size) / step_size
            step_start_weight = self.start_weight + (self.end_weight - self.start_weight) * (step / num_steps)
            step_end_weight = self.start_weight + (self.end_weight - self.start_weight) * ((step + 1) / num_steps)
            weight = step_start_weight + (step_end_weight - step_start_weight) * step_progress
        
        else:
            # Fallback to linear
            weight = self.start_weight + (self.end_weight - self.start_weight) * progress
        
        return float(np.clip(weight, 0.0, 1.0))
    
    def __call__(self, previous_state: State, new_state: State, action: np.ndarray) -> np.ndarray:
        """Compute the curriculum-shaped reward for each agent.
        
        Args:
            previous_state: Previous state of the environment.
            new_state: New state of the environment.
            action: Array of shape [num_agents, height, width] containing actions taken.
            
        Returns:
            np.ndarray: Reward for each agent, blended between shaped and minimal rewards.
        """
        # Compute both reward types
        shaped_reward = self.shaped_reward_fn(previous_state, new_state, action)
        minimal_reward = self.minimal_reward_fn(previous_state, new_state, action)
        
        # Get current curriculum weight
        weight = self._compute_weight(self.current_episode)
        
        # Blend rewards: R_t = w(t) · R_shaped + (1 − w(t)) · R_minimal
        reward = weight * shaped_reward + (1.0 - weight) * minimal_reward
        
        return reward.astype(np.float32)
    
    def get_current_weight(self) -> float:
        """Get the current curriculum weight w(t).
        
        Returns:
            Current weight for shaped reward blending.
        """
        return self._compute_weight(self.current_episode)

