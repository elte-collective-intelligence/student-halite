import numpy as np
from typing import Optional, Tuple, Any
from collections import deque
from pathlib import Path
from omegaconf import OmegaConf

from src.agents.agent import Agent
from src.env.constants import _NUM_ACTIONS, _DIRECTION_OFFSETS


# Sentinel object to detect if a parameter was not provided
_NOT_PROVIDED = object()


def _load_config_from_yaml() -> dict:
    """Load default configuration from yaml file."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "bots" / "rule_based.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        return OmegaConf.to_container(cfg, resolve=True)
    return {}


class RuleBasedBot(Agent):
    """Agent that either attacks adjacent enemies or moves towards nearest enemy,
    with movement strength threshold increasing as cells get closer to the border.
    Inner cells can combine forces to reach movement threshold.
    Can also prioritize moving towards high-production zones based on production_enemy_preference.
    """

    def __init__(
        self,
        agent_id: int,
        name: Any = _NOT_PROVIDED,
        base_strength: Any = _NOT_PROVIDED,
        scale: Any = _NOT_PROVIDED,
        max_search: Any = _NOT_PROVIDED,
        production_enemy_preference: Any = _NOT_PROVIDED,
    ):
        # Load default config from yaml if parameters are not specified
        default_config = _load_config_from_yaml()
        
        # Use yaml values as defaults, but allow explicit parameters to override
        name = name if name is not _NOT_PROVIDED else default_config.get("name", "RuleBased")
        base_strength = base_strength if base_strength is not _NOT_PROVIDED else default_config.get("base_strength", 50)
        scale = scale if scale is not _NOT_PROVIDED else default_config.get("scale", 10)
        max_search = max_search if max_search is not _NOT_PROVIDED else default_config.get("max_search", 4)
        production_enemy_preference = production_enemy_preference if production_enemy_preference is not _NOT_PROVIDED else default_config.get("production_enemy_preference", 0.5)
        
        super().__init__(agent_id, name)
        self.base_strength = base_strength
        self.scale = scale
        self.max_search = max_search  # max distance to search for enemies (optional)
        self.production_enemy_preference = production_enemy_preference  # 0 -> always enemies, 1 -> always production
        self.normalized_production = None  # Will be computed on first call
        self.production_computed = False
        self.last_grid_size = None  # Track grid size to detect changes

    def __call__(self, observation: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        ownership = observation[0]  # Ownership channel (1-based)
        strength = observation[1]  # Strength channel
        production = observation[2]  # Production channel
        height, width = ownership.shape
        agent_id_1b = self.agent_id + 1  # 1-based agent id

        # Initialize or recompute production map if grid size changed
        current_grid_size = (height, width)
        if not self.production_computed or self.last_grid_size != current_grid_size:
            self.normalized_production = self._compute_normalized_production(production, height, width)
            self.production_computed = True
            self.last_grid_size = current_grid_size

        # PHASE 1: Collect all desired moves (proposed actions)
        proposed_actions = self._collect_proposed_moves(
            ownership, strength, production, agent_id_1b, height, width, seed
        )

        # PHASE 2: Resolve conflicts and finalize actions
        actions = self._resolve_conflicts(
            ownership, strength, proposed_actions, agent_id_1b, height, width, seed
        )

        return actions

    def _collect_proposed_moves(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        production: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int,
        seed: Optional[int]
    ) -> np.ndarray:
        """Phase 1: Collect all desired moves without conflict resolution.
        Returns proposed actions array where each cell has its desired move.
        """
        # Initialize all actions to WAIT (0)
        proposed_actions = np.zeros((height, width), dtype=np.int32)

        # Step 1: Attack adjacent enemies or move into good neighbors
        self._attack_or_advance(
            ownership, strength, production, proposed_actions, agent_id_1b, height, width, seed
        )

        # Step 2: Identify frontier cells
        is_frontier = self._compute_frontier(ownership, agent_id_1b, height, width)

        # Step 3: Compute distance-to-frontier map
        dist_to_frontier = self._compute_distance_to_frontier(is_frontier, height, width)

        # Step 4: Move inner cells toward enemies
        self._move_inner_cells(
            ownership, strength, proposed_actions, is_frontier, dist_to_frontier,
            agent_id_1b, height, width, seed
        )

        # Step 5: Combine forces for non-moving cells (after all movement decisions)
        self._combine_forces_for_non_moving(
            ownership, strength, proposed_actions, is_frontier, dist_to_frontier,
            agent_id_1b, height, width
        )

        return proposed_actions

    def _resolve_conflicts(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        proposed_actions: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int,
        seed: Optional[int]
    ) -> np.ndarray:
        """Phase 2: Resolve movement conflicts and finalize actions.
        
        Conflicts can occur when:
        1. Multiple cells try to move to the same target (collision)
        2. Two cells swap positions (A->B, B->A)
        3. A cell tries to move to a cell that's also moving
        
        Resolution strategy:
        - For collisions: prioritize by strength (stronger cell wins)
        - For swaps: break the swap by keeping the stronger cell's move
        - For moving targets: cancel the move if target is also moving
        """
        rng = np.random.default_rng(seed)
        final_actions = np.zeros((height, width), dtype=np.int32)
        
        # Track which cells are moving and their targets
        # target_map[y, x] = (source_y, source_x) if cell (y, x) is targeted by source
        # Multiple sources can target the same cell (conflict)
        target_map = {}  # (ty, tx) -> list of (sy, sx, strength)
        
        # First pass: collect all proposed moves and their targets
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b:
                    continue
                
                action = proposed_actions[yi, xi]
                if action == 0:  # WAIT
                    continue
                
                # Calculate target position
                dy, dx = _DIRECTION_OFFSETS[action]
                ty, tx = (yi + dy) % height, (xi + dx) % width
                
                # Record this move
                if (ty, tx) not in target_map:
                    target_map[(ty, tx)] = []
                target_map[(ty, tx)].append((yi, xi, strength[yi, xi]))
        
        # Second pass: resolve conflicts
        # Track which cells have been finalized
        finalized = np.zeros((height, width), dtype=bool)
        
        # Pre-compute random tie-breakers for stable sorting
        tie_breakers = {}
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] == agent_id_1b:
                    tie_breakers[(yi, xi)] = rng.random()
        
        # Process each target cell
        for (ty, tx), sources in target_map.items():
            # Sort sources by strength (descending), then by tie-breaker
            sources_sorted = sorted(
                sources,
                key=lambda s: (s[2], tie_breakers.get((s[0], s[1]), 0.0)),
                reverse=True
            )
            
            # Winner is the strongest source (first in sorted list)
            winner_sy, winner_sx, winner_strength = sources_sorted[0]
            
            # Check if target is also moving (swap detection)
            target_action = proposed_actions[ty, tx]
            if target_action != 0 and ownership[ty, tx] == agent_id_1b:
                # Target is moving - check if it's a swap
                t_dy, t_dx = _DIRECTION_OFFSETS[target_action]
                t_ty, t_tx = (ty + t_dy) % height, (tx + t_dx) % width
                
                if t_ty == winner_sy and t_tx == winner_sx:
                    # Swap detected: A->B, B->A
                    # Break swap: stronger cell moves, weaker waits
                    target_strength = strength[ty, tx]
                    if winner_strength >= target_strength:
                        final_actions[winner_sy, winner_sx] = proposed_actions[winner_sy, winner_sx]
                        finalized[winner_sy, winner_sx] = True
                        finalized[ty, tx] = True  # Target waits
                    else:
                        final_actions[ty, tx] = proposed_actions[ty, tx]
                        finalized[ty, tx] = True
                        finalized[winner_sy, winner_sx] = True  # Winner waits
                    continue
            
            # No swap - check if winner can move to target
            # Can move if: target is neutral OR (target is ours and we're stronger)
            can_move = (
                ownership[ty, tx] != agent_id_1b or
                (ownership[ty, tx] == agent_id_1b and winner_strength > strength[ty, tx])
            )
            
            if can_move:
                final_actions[winner_sy, winner_sx] = proposed_actions[winner_sy, winner_sx]
                finalized[winner_sy, winner_sx] = True
            
            # All other sources wait (already 0 in final_actions)
            for sy, sx, _ in sources_sorted[1:]:
                finalized[sy, sx] = True
        
        # Third pass: finalize remaining cells that weren't in conflicts
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b:
                    continue
                if finalized[yi, xi]:
                    continue
                
                # Cell wasn't in any conflict, use proposed action
                final_actions[yi, xi] = proposed_actions[yi, xi]
        
        return final_actions

    def _compute_normalized_production(
        self,
        production: np.ndarray,
        height: int,
        width: int
    ) -> np.ndarray:
        """Compute normalized production map by summing 3x3 neighborhoods (torus) and normalizing."""
        # Sum production in 3x3 neighborhood for each cell (torus)
        normalized_prod = np.zeros_like(production, dtype=np.float32)
        
        for yi in range(height):
            for xi in range(width):
                # Sum 3x3 neighborhood with torus wrapping
                total = 0.0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        ny = (yi + ky) % height
                        nx = (xi + kx) % width
                        total += production[ny, nx]
                normalized_prod[yi, xi] = total
        
        # Normalize to [0, 1]
        prod_min = normalized_prod.min()
        prod_max = normalized_prod.max()
        if prod_max > prod_min:
            normalized_prod = (normalized_prod - prod_min) / (prod_max - prod_min)
        else:
            normalized_prod = np.zeros_like(normalized_prod)
        
        return normalized_prod

    def _attack_or_advance(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        production: np.ndarray,
        actions: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int,
        seed: Optional[int]
    ) -> None:
        """Attack adjacent enemies or move into cells that lead to enemies or high production zones."""
        max_search = self.max_search or (height + width)
        
        # Pre-compute normalized enemy distances for all directions from all cells
        enemy_distances = self._compute_enemy_distances(ownership, agent_id_1b, height, width, max_search)
        
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b or actions[yi, xi] != 0:
                    continue

                best_dir = self._find_best_direction(
                    ownership, strength, yi, xi, agent_id_1b, height, width,
                    enemy_distances
                )
                
                if best_dir != 0:
                    actions[yi, xi] = best_dir

    def _compute_enemy_distances(
        self,
        ownership: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int,
        max_search: int
    ) -> np.ndarray:
        """Pre-compute normalized enemy distances for efficiency."""
        # Shape: (height, width, num_directions)
        enemy_distances = np.full((height, width, _NUM_ACTIONS), max_search + 1, dtype=np.float32)
        
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b:
                    continue
                    
                for dir_idx in range(1, _NUM_ACTIONS):
                    dy, dx = _DIRECTION_OFFSETS[dir_idx]
                    ny, nx = (yi + dy) % height, (xi + dx) % width
                    
                    n_owner = ownership[ny, nx]
                    
                    if n_owner != 0 and n_owner != agent_id_1b:
                        # Immediate enemy
                        enemy_distances[yi, xi, dir_idx] = 1.0
                    else:
                        # Search in this direction for an enemy
                        dist = self._search_for_enemy(
                            ownership, ny, nx, dy, dx, agent_id_1b, height, width, max_search
                        )
                        enemy_distances[yi, xi, dir_idx] = float(dist)
        
        # Normalize distances to [0, 1] (lower is better, so we invert)
        # Find min and max across all valid distances
        valid_distances = enemy_distances[enemy_distances <= max_search]
        if len(valid_distances) > 0:
            dist_min = valid_distances.min()
            dist_max = valid_distances.max()
            if dist_max > dist_min:
                # Normalize: (max - dist) / (max - min), so closer enemies have higher scores
                enemy_distances = np.where(
                    enemy_distances <= max_search,
                    (dist_max - enemy_distances) / (dist_max - dist_min),
                    0.0
                )
        
        return enemy_distances

    def _find_best_direction(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        yi: int,
        xi: int,
        agent_id_1b: int,
        height: int,
        width: int,
        enemy_distances: np.ndarray
    ) -> int:
        """Find the best direction considering both enemies and production based on preference."""
        best_dir = 0
        best_combined_score = -np.inf
        
        for dir_idx in range(1, _NUM_ACTIONS):
            dy, dx = _DIRECTION_OFFSETS[dir_idx]
            ny, nx = (yi + dy) % height, (xi + dx) % width

            n_owner = ownership[ny, nx]
            n_strength = strength[ny, nx]

            # Only consider non-self neighbors that we can beat
            if n_owner == agent_id_1b or strength[yi, xi] <= n_strength:
                continue

            # Get normalized enemy distance score (already normalized)
            enemy_score = enemy_distances[yi, xi, dir_idx]
            
            # Get normalized production score
            production_score = self.normalized_production[ny, nx]
            
            # Combine scores based on preference
            # production_enemy_preference = 0 -> always enemies (weight 1.0)
            # production_enemy_preference = 1 -> always production (weight 1.0)
            combined_score = (
                self.production_enemy_preference * production_score +
                (1.0 - self.production_enemy_preference) * enemy_score
            )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_dir = dir_idx

        return best_dir


    def _search_for_enemy(
        self,
        ownership: np.ndarray,
        start_y: int,
        start_x: int,
        dy: int,
        dx: int,
        agent_id_1b: int,
        height: int,
        width: int,
        max_search: int
    ) -> int:
        """Search in a direction for an enemy cell."""
        score = 1
        cy, cx = start_y, start_x
        found_enemy = False

        while score < max_search:
            cy = (cy + dy) % height
            cx = (cx + dx) % width
            score += 1
            cur_owner = ownership[cy, cx]

            if cur_owner != 0 and cur_owner != agent_id_1b:
                found_enemy = True
                break
            if cur_owner == agent_id_1b:
                found_enemy = False
                break

        return score if found_enemy else max_search + 1

    def _compute_frontier(
        self,
        ownership: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int
    ) -> np.ndarray:
        """Identify cells adjacent to non-owned territory."""
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        is_frontier = np.zeros_like(ownership, dtype=bool)

        for dir_idx in range(1, _NUM_ACTIONS):
            dy, dx = _DIRECTION_OFFSETS[dir_idx]
            ny, nx = (y + dy) % height, (x + dx) % width
            is_frontier |= (ownership[ny, nx] != agent_id_1b)

        return is_frontier

    def _compute_distance_to_frontier(
        self,
        is_frontier: np.ndarray,
        height: int,
        width: int
    ) -> np.ndarray:
        """Compute distance map from each cell to the nearest frontier cell using BFS."""
        INF = 10**9
        dist_to_frontier = np.full((height, width), INF, dtype=np.int32)
        q = deque()

        # Initialize frontier cells
        for yi in range(height):
            for xi in range(width):
                if is_frontier[yi, xi]:
                    dist_to_frontier[yi, xi] = 0
                    q.append((yi, xi))

        # BFS to compute distances
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while q:
            cy, cx = q.popleft()
            for dy, dx in dirs:
                ny, nx = (cy + dy) % height, (cx + dx) % width
                if dist_to_frontier[ny, nx] > dist_to_frontier[cy, cx] + 1:
                    dist_to_frontier[ny, nx] = dist_to_frontier[cy, cx] + 1
                    q.append((ny, nx))

        return dist_to_frontier

    def _move_inner_cells(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        actions: np.ndarray,
        is_frontier: np.ndarray,
        dist_to_frontier: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int,
        seed: Optional[int]
    ) -> None:
        """Move inner cells toward frontier, prioritizing high production areas or enemies based on production_enemy_preference."""
        rng = np.random.default_rng(seed)
        max_d = np.max(dist_to_frontier)
        max_search = self.max_search or (height + width)
        
        # Pre-compute normalized enemy distances for all directions from all cells
        enemy_distances = self._compute_enemy_distances(ownership, agent_id_1b, height, width, max_search)

        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b or actions[yi, xi] != 0:
                    continue

                d = dist_to_frontier[yi, xi]
                required_strength = np.min([self.base_strength + self.scale * (max_d - d), 255])
                current_strength = strength[yi, xi]

                if current_strength == 0:
                    actions[yi, xi] = 0
                    continue

                # If strong enough, move toward frontier
                if current_strength >= required_strength:
                    if not is_frontier[yi, xi]:
                        best_dir = self._find_direction_to_frontier_with_preference(
                            ownership, strength, yi, xi, agent_id_1b, height, width,
                            dist_to_frontier, enemy_distances
                        )
                        actions[yi, xi] = best_dir
                    continue

    def _find_direction_to_frontier_with_preference(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        yi: int,
        xi: int,
        agent_id_1b: int,
        height: int,
        width: int,
        dist_to_frontier: np.ndarray,
        enemy_distances: np.ndarray
    ) -> int:
        """Find the best direction towards frontier, considering both production and enemies.
        Only considers directions that move closer to the frontier.
        Uses production_enemy_preference to weight between enemies (0) and production (1).
        """
        best_dir = 0
        best_combined_score = -np.inf
        current_dist = dist_to_frontier[yi, xi]

        for dir_idx in range(1, _NUM_ACTIONS):
            dy, dx = _DIRECTION_OFFSETS[dir_idx]
            ny, nx = (yi + dy) % height, (xi + dx) % width

            # Only consider directions that move us closer to the frontier
            neighbor_dist = dist_to_frontier[ny, nx]
            if neighbor_dist >= current_dist:
                continue

            # Only consider neighbors we can move into (not stronger enemies)
            n_owner = ownership[ny, nx]
            n_strength = strength[ny, nx]
            if n_owner != agent_id_1b and strength[yi, xi] <= n_strength:
                continue

            # Get normalized enemy distance score (already normalized, higher is better)
            enemy_score = enemy_distances[yi, xi, dir_idx]
            
            # Get normalized production score at the neighbor
            production_score = self.normalized_production[ny, nx]
            
            # Combine scores based on preference
            # production_enemy_preference = 0 -> always enemies (weight 1.0)
            # production_enemy_preference = 1 -> always production (weight 1.0)
            combined_score = (
                self.production_enemy_preference * production_score +
                (1.0 - self.production_enemy_preference) * enemy_score
            )
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_dir = dir_idx

        # If no valid direction found (shouldn't happen, but fallback to original method)
        if best_dir == 0:
            return self._find_direction_to_frontier(ownership, yi, xi, agent_id_1b, height, width)
        
        return best_dir

    def _find_direction_to_frontier(
        self,
        ownership: np.ndarray,
        yi: int,
        xi: int,
        agent_id_1b: int,
        height: int,
        width: int
    ) -> int:
        """Find the direction that leads most quickly to non-owned territory (fallback method)."""
        best_dir = 0
        min_dist = height + width

        for dir_idx in range(1, _NUM_ACTIONS):
            dy, dx = _DIRECTION_OFFSETS[dir_idx]
            ny, nx = (yi + dy) % height, (xi + dx) % width

            dist = 1
            cy, cx = ny, nx
            while ownership[cy, cx] == agent_id_1b and dist < min_dist:
                cy = (cy + dy) % height
                cx = (cx + dx) % width
                dist += 1

            if dist < min_dist:
                min_dist = dist
                best_dir = dir_idx

        return best_dir

    def _combine_forces_for_non_moving(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        actions: np.ndarray,
        is_frontier: np.ndarray,
        dist_to_frontier: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int
    ) -> None:
        """Combine forces for cells that are not moving after all movement decisions."""
        max_d = np.max(dist_to_frontier)
        combined_mask = np.zeros_like(ownership, dtype=bool)
        max_strength = 255

        # First pass: collect all non-moving cells that need to combine
        for yi in range(height):
            for xi in range(width):
                # Only consider our cells that are not moving and not on frontier
                if (ownership[yi, xi] != agent_id_1b or 
                    actions[yi, xi] != 0 or 
                    is_frontier[yi, xi] or 
                    combined_mask[yi, xi] or
                    strength[yi, xi] == 0):
                    continue

                d = dist_to_frontier[yi, xi]
                required_strength = self.base_strength + self.scale * (max_d - d)
                current_strength = strength[yi, xi]

                # Only try to combine if not strong enough
                if current_strength < required_strength:
                    self._try_combine_forces(
                        ownership, strength, actions, combined_mask, is_frontier,
                        yi, xi, required_strength, agent_id_1b, height, width, max_strength
                    )

    def _try_combine_forces(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        actions: np.ndarray,
        combined_mask: np.ndarray,
        is_frontier: np.ndarray,
        yi: int,
        xi: int,
        required_strength: float,
        agent_id_1b: int,
        height: int,
        width: int,
        max_strength: int
    ) -> None:
        """Try to combine forces with a neighbor to meet movement threshold.
        Only considers neighbors that are also not moving."""
        for dir_idx in range(1, _NUM_ACTIONS):
            dy, dx = _DIRECTION_OFFSETS[dir_idx]
            ny, nx = (yi + dy) % height, (xi + dx) % width

            # Neighbor must be: our cell, not moving, not frontier, not already combined
            if (ownership[ny, nx] == agent_id_1b and 
                actions[ny, nx] == 0 and
                not combined_mask[ny, nx] and 
                not is_frontier[ny, nx] and
                strength[ny, nx] > 0):
                
                combined_strength = strength[yi, xi] + strength[ny, nx]

                if combined_strength >= required_strength and combined_strength <= max_strength:
                    # Mark both as combined
                    combined_mask[yi, xi] = True
                    combined_mask[ny, nx] = True

                    # Move combined force
                    actions[yi, xi] = dir_idx
                    actions[ny, nx] = 0
                    break
