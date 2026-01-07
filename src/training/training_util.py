"""Utility functions for training multi-agent RL algorithms on Halite."""

import numpy as np
from typing import Tuple, Dict, List


def extract_local_observations(
    grid: np.ndarray,
    agent_id: int,
    patch_size: int = 7
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract local 7x7 observations for each unit owned by the agent.
    Vectorized implementation using numpy for speed.
    
    Args:
        grid: Global grid observation of shape (3, H, W) with channels:
              [owner, strength, production]
        agent_id: Agent ID (0-indexed)
        patch_size: Size of the local patch (default 7)
    
    Returns:
        local_obs: Array of shape (num_units, patch_size, patch_size, 6) containing
                  local observations for each unit
        unit_positions: List of (y, x) positions for each unit
    """
    owner = grid[0]  # (H, W)
    strength = grid[1]  # (H, W)
    production = grid[2]  # (H, W)
    
    H, W = owner.shape
    agent_id_1b = agent_id + 1  # Convert to 1-based
    
    # Find all cells owned by this agent
    agent_mask = (owner == agent_id_1b)
    unit_y, unit_x = np.where(agent_mask)
    num_units = len(unit_y)
    
    if num_units == 0:
        return np.zeros((0, patch_size, patch_size, 6), dtype=np.float32), []
    
    half_patch = patch_size // 2
    
    # Create offset grids for patch coordinates
    # py, px: (patch_size, patch_size) - patch coordinates relative to center
    py_grid, px_grid = np.meshgrid(
        np.arange(patch_size) - half_patch,
        np.arange(patch_size) - half_patch,
        indexing='ij'
    )
    # py_grid, px_grid: (patch_size, patch_size)
    
    # Broadcast to all units: (num_units, patch_size, patch_size)
    # Compute global coordinates with toroidal wrapping
    unit_y_expanded = unit_y[:, None, None]  # (num_units, 1, 1)
    unit_x_expanded = unit_x[:, None, None]  # (num_units, 1, 1)
    
    gy = (unit_y_expanded + py_grid) % H  # (num_units, patch_size, patch_size)
    gx = (unit_x_expanded + px_grid) % W  # (num_units, patch_size, patch_size)
    
    # Flatten for advanced indexing
    gy_flat = gy.ravel()  # (num_units * patch_size * patch_size,)
    gx_flat = gx.ravel()  # (num_units * patch_size * patch_size,)
    
    # Extract all values at once using advanced indexing
    owner_patches = owner[gy_flat, gx_flat].reshape(num_units, patch_size, patch_size)
    strength_patches = strength[gy_flat, gx_flat].reshape(num_units, patch_size, patch_size)
    production_patches = production[gy_flat, gx_flat].reshape(num_units, patch_size, patch_size)
    
    # Normalize strength and production to [0, 1]
    max_strength = 255.0  # From constants
    max_production = np.max(production) if np.max(production) > 0 else 1.0
    
    # Initialize output array
    local_obs = np.zeros((num_units, patch_size, patch_size, 6), dtype=np.float32)
    
    # Channel 0: is_mine (1 if owned by current agent, 0 otherwise)
    local_obs[:, :, :, 0] = (owner_patches == agent_id_1b).astype(np.float32)
    
    # Channel 1: is_enemy (1 if owned by enemy, 0 otherwise)
    is_enemy = (owner_patches > 0) & (owner_patches != agent_id_1b)
    local_obs[:, :, :, 1] = is_enemy.astype(np.float32)
    
    # Channel 2: is_neutral (1 if neutral, 0 otherwise)
    local_obs[:, :, :, 2] = (owner_patches == 0).astype(np.float32)
    
    # Channel 3: normalized_strength [0, 1]
    local_obs[:, :, :, 3] = strength_patches.astype(np.float32) / max_strength
    
    # Channel 4: normalized_production [0, 1]
    local_obs[:, :, :, 4] = production_patches.astype(np.float32) / max_production
    
    # Channel 5: unit_mask (1 if this cell corresponds to the current unit, 0 otherwise)
    # Create coordinate grids for unit positions
    unit_y_grid = np.broadcast_to(unit_y[:, None, None], (num_units, patch_size, patch_size))
    unit_x_grid = np.broadcast_to(unit_x[:, None, None], (num_units, patch_size, patch_size))
    unit_mask = (gy == unit_y_grid) & (gx == unit_x_grid)
    local_obs[:, :, :, 5] = unit_mask.astype(np.float32)
    
    unit_positions = [(int(y), int(x)) for y, x in zip(unit_y, unit_x)]
    return local_obs, unit_positions


def create_local_observation_dict(
    obs: Dict[str, np.ndarray],
    num_agents: int
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Create local observations for all agents.
    
    Args:
        obs: Global observation dictionary with 'grid', 'action_mask', 'step_count'
        num_agents: Number of agents
    
    Returns:
        Dictionary mapping agent_id to their local observations:
        {
            agent_id: {
                'local_obs': (num_units, 7, 7, 6),
                'unit_positions': [(y, x), ...],
                'action_mask': (num_units, 5) - action masks for each unit
            }
        }
    """
    grid = obs['grid']  # (3, H, W)
    action_mask = obs['action_mask']  # (num_agents, H, W, 5)
    
    agent_observations = {}
    
    for agent_id in range(num_agents):
        local_obs, unit_positions = extract_local_observations(grid, agent_id)
        
        # Extract action masks for each unit
        agent_action_mask = action_mask[agent_id]  # (H, W, 5)
        unit_action_masks = []
        for y, x in unit_positions:
            unit_action_masks.append(agent_action_mask[y, x])  # (5,)
        
        unit_action_masks = np.array(unit_action_masks) if unit_action_masks else np.zeros((0, 5), dtype=bool)
        
        agent_observations[agent_id] = {
            'local_obs': local_obs,
            'unit_positions': unit_positions,
            'action_mask': unit_action_masks
        }
    
    return agent_observations




