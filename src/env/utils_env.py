from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from typing import NamedTuple

from src.env.constants import _MAX_STRENGTH, _NEW_CELL_STRENGTH, _DIRECTION_OFFSETS, _DAMAGE_KERNEL, _NUM_ACTIONS
from src.env.types import State, Observation, Action

def update_game_state(state: State, player_moves: np.ndarray) -> State:
    """
    Advance the game state by one turn based on player moves.
    
    Args:
        state: Current game state containing grid, alive status, and turn count
        player_moves: Array of shape [num_players, height, width] with movement directions
        
    Returns:
        New game state after processing moves, damage, and ownership changes
    """

    strength_maps, ownership_maps, neutral_damage = _process_player_moves(state.grid, state.alive, player_moves)
    damage_maps = _calculate_damage_to_players(state.grid, state.alive, strength_maps)
    updated_strengths = _apply_damage_to_players(strength_maps, damage_maps)
    new_grid = _update_grid(state.grid, updated_strengths, ownership_maps, damage_maps, neutral_damage)
    new_alive = _check_player_survival(new_grid, state.alive)
    action_mask = _get_action_mask(new_grid, state.alive.shape[0])
    
    return State(
        grid=new_grid,
        alive=new_alive,
        step_count=state.step_count + 1,
        action_mask=action_mask
    )

def _process_player_moves(grid: np.ndarray, alive: np.ndarray,
                        player_moves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Process all player moves and compute resulting strengths and ownership.
    
    Args:
        grid: Current game grid [channels, height, width]
        alive: Boolean array [num_players] of active players
        player_moves: Array [num_players, height, width] of movement directions
        
    Returns:
        Tuple containing:
        - strength_maps: Array [num_players, height, width] of updated strengths
        - ownership_maps: Array [num_players, height, width] of claimed cells
        - neutral_damage: Array [height, width] of damage to neutral cells
    """

    num_players = alive.shape[0]
    height, width = grid.shape[1], grid.shape[2]
    
    # Precompute all target positions for each direction [5, height, width, 2]
    def get_target_positions():
        y_indices = np.arange(height)
        x_indices = np.arange(width)
        yy, xx = np.meshgrid(y_indices, x_indices, indexing='ij')
        
        target_yy = (yy[None,...] + _DIRECTION_OFFSETS[:,0,None,None]) % height
        target_xx = (xx[None,...] + _DIRECTION_OFFSETS[:,1,None,None]) % width
        return np.stack([target_yy, target_xx], axis=-1)
    
    target_positions = get_target_positions()

    # Create mask of valid moves
    player_ownership = grid[0]  # [height, width]
    player_ids = np.arange(1, num_players + 1)
    owned_mask = (player_ownership == player_ids[:, None, None])
    valid_mask = owned_mask & alive[:, None, None]
    source_strengths = np.where(valid_mask, grid[1], 0)

    def accumulate_for_player(player_idx):
        player_moves_dir = player_moves[player_idx]
        player_strengths = source_strengths[player_idx]
        
        # - STILL pieces -
        still_mask = (player_moves_dir == 0) & valid_mask[player_idx]
        production = grid[2]
        new_strength = np.minimum(
            player_strengths.astype(np.int32) + np.where(still_mask, production, 0).astype(np.int32),
            _MAX_STRENGTH
        ).astype(np.uint8)

        # - MOVING pieces -

        # Set source to NEW_CELL_STRENGTH
        moving_mask = (player_moves_dir != 0) & valid_mask[player_idx]
        moving_strengths = np.where(moving_mask, player_strengths, 0)
        new_strength = np.where(moving_mask, _NEW_CELL_STRENGTH, new_strength)
        
        # Initialize empty pieces and ownership grids
        player_pieces = np.zeros((height, width), dtype=np.int32)
        player_own_map = np.zeros((height, width), dtype=bool)
        moving_pieces = np.zeros((height, width), dtype=np.int32)  # Track moved pieces at targets
        
        # Add NEW pieces to their original positions
        player_pieces = player_pieces + new_strength
        # Preserve ownership for cells where player stays, even if strength is 0
        player_own_map = (player_strengths > 0) | (new_strength > 0) | still_mask
        
        # Process all directions
        for dir_idx in range(5):
            dir_mask = (player_moves_dir == dir_idx) & valid_mask[player_idx] & (dir_idx != 0)  # Skip direction 0 (STILL)
            target_yy = target_positions[dir_idx, :, :, 0]
            target_xx = target_positions[dir_idx, :, :, 1]
            
            moved_strengths = np.where(dir_mask, player_strengths, 0).astype(np.uint8)
            
            # Updates ownership claims
            player_own_map[target_yy, target_xx] = player_own_map[target_yy, target_xx] | dir_mask

            # Add the moved strength to the corresponding target position
            player_pieces[target_yy, target_xx] += moved_strengths

            # Accumulates strength values moving into each cell (used to calculate damage to neutral cells)
            moving_pieces[target_yy, target_xx] += moved_strengths

        # Ensure origin cell is set to NEW_CELL_STRENGTH and owned if a move was made
        # player_pieces = np.where(moving_mask, _NEW_CELL_STRENGTH, player_pieces)
        player_own_map = np.where(moving_mask, True, player_own_map)

        return (np.minimum(player_pieces, _MAX_STRENGTH).astype(np.uint8), player_own_map, moving_pieces)
    
    # Process each player
    all_pieces = []
    all_own_maps = []
    all_moving_strengths = []
    for player_idx in range(num_players):
        pieces, own_map, moving_strengths = accumulate_for_player(player_idx)
        all_pieces.append(pieces)
        all_own_maps.append(own_map)
        all_moving_strengths.append(moving_strengths)
    
    return np.array(all_pieces), np.array(all_own_maps), np.sum(all_moving_strengths, axis=0)

def _calculate_damage_to_players(grid: np.ndarray, alive: np.ndarray, pieces: np.ndarray) -> np.ndarray:
    """
    Compute damage each player's pieces would receive from enemies and neutrals.
    
    Args:
        grid: Current game grid [channels, height, width]
        alive: Boolean array [num_players] of active players
        pieces: Array [num_players, height, width] of player strengths
        
    Returns:
        damage: Array [num_players, height, width] of damage values
    """
    num_players = alive.shape[0]

    def damage_for_player(player_idx):

        # Get all enemy indices
        enemy_indices = (np.arange(num_players) != player_idx) & alive
        
        # Sum enemy strengths
        enemy_strengths = np.sum(
            np.where(enemy_indices[:, None, None], pieces.astype(np.float32), 0),
            axis=0
        )
        
        # Perform convolution with circular padding using scipy
        damage = ndimage.convolve(enemy_strengths, _DAMAGE_KERNEL, mode='wrap')
        
        # Add damage from neutral cells
        neutral_damage = np.where(
            (grid[0] == 0) & (grid[1] > 0),
            grid[1].astype(np.float32),
            0.0
        )
        
        # Cap damage at MAX_STRENGTH before casting to uint8 to prevent overflow
        total_damage = damage + neutral_damage
        return np.minimum(total_damage, _MAX_STRENGTH).astype(np.uint8)

    # Process each player
    damages = []
    for player_idx in range(num_players):
        damages.append(damage_for_player(player_idx))
    return np.array(damages)

def _apply_damage_to_players(pieces: np.ndarray, damage: np.ndarray) -> np.ndarray:
    """
    Apply damage to pieces and clip resulting strengths.
    
    Args:
        pieces: Array [num_players, height, width] of current strengths
        damage: Array [num_players, height, width] of damage values
        
    Returns:
        Array [num_players, height, width] of updated strengths after damage
    """
    return np.maximum(pieces.astype(np.int32) - damage.astype(np.int32), 0).astype(np.uint8)

def _update_grid(grid: np.ndarray, strengths: np.ndarray, ownership: np.ndarray, damages: np.ndarray, neutral_damage: np.ndarray) -> np.ndarray:
    """
    Resolve conflicts and update grid with final ownership and strengths.
    
    Args:
        grid: Current game grid [channels, height, width]
        strengths: Array [num_players, height, width] of post-damage strengths
        ownership: Array [num_players, height, width] of ownership claims
        damages: Array [num_players, height, width] of damage values
        neutral_damage: Array [height, width] of damage to neutral cells
        
    Returns:
        New game grid [channels, height, width] with updated state
    """
    
    # Find max strength and potential new owner for each cell
    strength = np.max(strengths, axis=0)
    potential_new_owner = np.argmax(strengths, axis=0) + 1  # +1 to match player numbering
    potential_new_owner = np.where(strength == 0, 0, potential_new_owner)
    
    # Resolves conflicts in ownership -> cells claimed by exactly one player go to that player
    ownership = _combine_ownership_maps(ownership)
    
    # Get damage for current owner (convert ownership to 0-based index for damages array)
    owner_index = ownership - 1
    valid_owner_mask = owner_index >= 0  # Or owner_index > 0 if you want to exclude player 0

    # Initialize with zeros
    owner_damage = np.zeros_like(owner_index, dtype=damages.dtype)
    # Use advanced indexing to get damage for each owner
    height, width = owner_index.shape
    y_indices, x_indices = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    valid_mask = valid_owner_mask & (owner_index < damages.shape[0])
    owner_damage[valid_mask] = damages[owner_index[valid_mask], y_indices[valid_mask], x_indices[valid_mask]]

    # If a cell was previously owned and took no damage, keep the original owner
    #   -> (this case handles the new pieces caused by MOVING pieces)
    # Otherwise, give ownership to the player with maximum strength
    # If no players have strength (all zeros), cell becomes neutral
    owner = np.where(
        (ownership > 0) & (owner_damage == 0),
        ownership,
        potential_new_owner
    )

    current_strength = grid[1].astype(np.int32)
    damage = neutral_damage.astype(np.int32)
    neutral_mask = (grid[0] == 0)

    # If remaining neutral: apply damage from moving pieces
    # If being conquered: use the conquering player's strength
    final_strength = np.where(
        neutral_mask & (owner == 0),  # Still neutral
        np.maximum(current_strength - damage, 0),
        strength.astype(np.int32)  # Use new strength
    ).astype(np.uint8)
    
    # Keep production from original grid
    production = grid[2]
    
    # Zero-strength mutual destruction logic
    zero_owned = (owner > 0) & (final_strength == 0)

    # Use only the four movement directions (skip STILL)
    neighbor_shifts = _DIRECTION_OFFSETS[1:]
    to_neutralize = np.zeros_like(zero_owned, dtype=bool)
    for shift in neighbor_shifts:
        dy, dx = shift
        neighbor_zero = np.roll(zero_owned, shift=(dy, dx), axis=(0,1))
        neighbor_owner = np.roll(owner, shift=(dy, dx), axis=(0,1))
        conflict = zero_owned & neighbor_zero & (owner != neighbor_owner) & (neighbor_owner > 0)
        to_neutralize = to_neutralize | conflict

    owner = np.where(to_neutralize, 0, owner)
    final_strength = np.where(to_neutralize, 0, final_strength)

    return np.stack([owner, final_strength, production])

def _check_player_survival(grid: np.ndarray, alive: np.ndarray) -> np.ndarray:
    """
    Determine which players still have remaining pieces on the board.
    
    Args:
        grid: Current game grid [channels, height, width]
        alive: Boolean array [num_players] of current player status
        
    Returns:
        Updated alive status array [num_players] after survival check
    """
    num_players = alive.shape[0]
    owners = grid[0]
    
    has_pieces = np.array([np.any(owners == (player_idx + 1)) for player_idx in range(num_players)])
    
    return has_pieces & alive

def _combine_ownership_maps(ownership_maps: np.ndarray) -> np.ndarray:
    """
    Resolve conflicting ownership claims into a single grid.
    
    Args:
        ownership_maps: Boolean array [num_players, height, width] of claims
        
    Returns:
        Integer array [height, width] where:
        - 0: No owner or conflicting claims
        - 1..N: Single player owns the cell
    """

    # Convert bool to player indices (0 means not owned)
    player_indices = np.arange(1, ownership_maps.shape[0] + 1)  # 1-based player indices
    weighted_maps = ownership_maps * player_indices[:, None, None]
    
    # Sum across players to detect conflicts
    ownership_sum = np.sum(weighted_maps, axis=0)
    
    # Count how many players claim each cell
    claim_count = np.sum(ownership_maps.astype(np.int32), axis=0)
    
    # Only keep cells with exactly one claim
    combined_ownership = np.where(claim_count == 1, ownership_sum, 0)
    
    return combined_ownership.astype(np.int32)

def _get_action_mask(grid: np.ndarray, num_agents: int) -> np.ndarray:
    """
    Get the action mask for the current state.
    
    Args:
        grid: The environment grid of shape [channels, rows, cols]
        
    Returns:
        np.ndarray: Action mask of shape [num_agents, rows, cols, num_actions] where:
                    - True means the action is allowed
                    - False means the action is not allowed
                    Each agent can only act on cells they own (grid[0] == agent_id + 1)
    """
   
    ownership = grid[0]
    agent_masks = np.arange(1, num_agents + 1)[:, None, None] == ownership
    action_mask = agent_masks[..., None] * np.ones(_NUM_ACTIONS, dtype=bool)
    
    return action_mask