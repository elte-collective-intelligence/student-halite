"""Replay buffer for off-policy RL algorithms."""

import numpy as np
from collections import deque
from typing import Optional, Tuple, Dict, Any
import torch


class ReplayBuffer:
    """Experience replay buffer for Q-learning algorithms."""
    
    def __init__(
        self,
        capacity: int = 100000,
        num_agents: int = 2,
        grid_size: Tuple[int, int] = (5, 5),
        num_actions: int = 5
    ):
        self.capacity = capacity
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.num_actions = num_actions
        
        # Track actual observed grid dimensions (may differ from grid_size due to generator adjustments)
        self.actual_grid_size = None
        
        self.buffer = deque(maxlen=capacity)
        
    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray
    ):
        """Add a transition to the buffer.
        
        Args:
            obs: (channels, height, width)
            action: (num_agents, height, width)
            reward: (num_agents,)
            next_obs: (channels, height, width)
            done: bool
            action_mask: (num_agents, height, width, num_actions)
            next_action_mask: (num_agents, height, width, num_actions)
        """
        # Detect or validate actual grid size
        obs_h, obs_w = obs.shape[1], obs.shape[2]
        
        if self.actual_grid_size is None:
            # First observation - set the actual grid size
            self.actual_grid_size = (obs_h, obs_w)
        
        # Only store if shapes match the actual grid size we're tracking
        expected_obs_shape = (3, self.actual_grid_size[0], self.actual_grid_size[1])
        expected_action_shape = (self.num_agents, self.actual_grid_size[0], self.actual_grid_size[1])
        expected_mask_shape = (self.num_agents, self.actual_grid_size[0], self.actual_grid_size[1], self.num_actions)
        
        if (obs.shape != expected_obs_shape or 
            next_obs.shape != expected_obs_shape or
            action.shape != expected_action_shape or
            action_mask.shape != expected_mask_shape or
            next_action_mask.shape != expected_mask_shape):
            # Skip storing if shapes don't match
            return
        
        # Store copies to avoid reference issues
        self.buffer.append({
            'obs': obs.copy(),
            'action': action.copy(),
            'reward': reward.copy(),
            'next_obs': next_obs.copy(),
            'done': done,
            'action_mask': action_mask.copy(),
            'next_action_mask': next_action_mask.copy()
        })
    
    def sample(
        self,
        batch_size: int,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions.
        
        Returns:
            Dictionary with batched tensors
        """
        # Use replace=True if buffer is smaller than batch_size, otherwise False
        replace = batch_size > len(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=replace)
        batch = [self.buffer[idx] for idx in indices]
        
        # Check that all items have consistent shapes
        obs_shapes = [b['obs'].shape for b in batch]
        action_shapes = [b['action'].shape for b in batch]
        
        if len(set(obs_shapes)) > 1 or len(set(action_shapes)) > 1:
            # Debug: print shapes if they don't match
            print(f"Warning: Inconsistent shapes in replay buffer batch:")
            print(f"  Obs shapes: {set(obs_shapes)}")
            print(f"  Action shapes: {set(action_shapes)}")
            # Filter to only use items with the most common shape
            from collections import Counter
            most_common_obs_shape = Counter(obs_shapes).most_common(1)[0][0]
            most_common_action_shape = Counter(action_shapes).most_common(1)[0][0]
            batch = [
                b for b in batch
                if b['obs'].shape == most_common_obs_shape and b['action'].shape == most_common_action_shape
            ]
            if len(batch) == 0:
                raise ValueError("No items with consistent shapes found in replay buffer")
        
        obs_batch = torch.FloatTensor(
            np.stack([b['obs'] for b in batch], axis=0)
        ).to(device)
        action_batch = torch.LongTensor(
            np.stack([b['action'] for b in batch], axis=0)
        ).to(device)
        reward_batch = torch.FloatTensor(
            np.stack([b['reward'] for b in batch], axis=0)
        ).to(device)
        next_obs_batch = torch.FloatTensor(
            np.stack([b['next_obs'] for b in batch], axis=0)
        ).to(device)
        done_batch = torch.BoolTensor(
            np.array([b['done'] for b in batch])
        ).to(device)
        action_mask_batch = torch.BoolTensor(
            np.stack([b['action_mask'] for b in batch], axis=0)
        ).to(device)
        next_action_mask_batch = torch.BoolTensor(
            np.stack([b['next_action_mask'] for b in batch], axis=0)
        ).to(device)
        
        return {
            'obs': obs_batch,
            'action': action_batch,
            'reward': reward_batch,
            'next_obs': next_obs_batch,
            'done': done_batch,
            'action_mask': action_mask_batch,
            'next_action_mask': next_action_mask_batch
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class EpisodeBuffer:
    """Buffer for on-policy algorithms (PPO) that stores entire episodes."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the buffer for a new episode."""
        self.obs = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        done: bool,
        action_mask: np.ndarray
    ):
        """Add a step to the buffer."""
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)
    
    def get(
        self,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """Convert buffer to tensors.
        
        Returns:
            Dictionary with episode tensors
        """
        return {
            'obs': torch.FloatTensor(np.array(self.obs)).to(device),
            'actions': torch.LongTensor(np.array(self.actions)).to(device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).to(device),
            'values': torch.FloatTensor(np.array(self.values)).to(device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(device),
            'dones': torch.BoolTensor(np.array(self.dones)).to(device),
            'action_masks': torch.BoolTensor(np.array(self.action_masks)).to(device)
        }

