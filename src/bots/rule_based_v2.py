import numpy as np
from typing import Optional
from collections import deque

from src.agents.agent import Agent
from src.env.constants import _NUM_ACTIONS, _DIRECTION_OFFSETS


class RuleBasedV2(Agent):
    """Variation of RuleBasedBot2 that prioritizes high-production zones.
    Uses lower strength thresholds and prioritizes expansion over combat.
    """

    def __init__(
        self,
        agent_id: int,
        name: str = "RB_V2",
        base_strength: int = 30,
        scale: int = 5,
        production_threshold: float = 0.6,
    ):
        super().__init__(agent_id, name)
        self.base_strength = base_strength  # Lower threshold for faster movement
        self.scale = scale  # Smaller scale for less strict requirements
        self.production_threshold = production_threshold  # Threshold for high-production zones
        self.normalized_production = None
        self.production_computed = False

    def __call__(self, observation: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        ownership = observation[0]  # Ownership channel (1-based)
        strength = observation[1]  # Strength channel
        production = observation[2]  # Production channel
        height, width = ownership.shape
        agent_id_1b = self.agent_id + 1  # 1-based agent id

        # Initialize production map on first call
        if not self.production_computed:
            self.normalized_production = self._compute_normalized_production(production, height, width)
            self.production_computed = True

        # Initialize all actions to WAIT (0)
        actions = np.zeros((height, width), dtype=np.int32)

        # Step 1: Attack adjacent enemies (only if weak enemies)
        self._attack_weak_adjacent_enemies(
            ownership, strength, actions, agent_id_1b, height, width
        )

        # Step 2: Expand into high-production neutral zones
        self._expand_to_high_production(
            ownership, strength, actions, agent_id_1b, height, width
        )

        # Step 3: Identify frontier cells
        is_frontier = self._compute_frontier(ownership, agent_id_1b, height, width)

        # Step 4: Move toward frontier (lower threshold, prioritize production)
        self._move_toward_frontier_with_production(
            ownership, strength, actions, is_frontier, agent_id_1b, height, width, seed
        )

        return actions

    def _compute_normalized_production(
        self,
        production: np.ndarray,
        height: int,
        width: int
    ) -> np.ndarray:
        """Compute normalized production map by summing 3x3 neighborhoods (torus) and normalizing."""
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

    def _attack_weak_adjacent_enemies(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        actions: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int
    ) -> None:
        """Attack adjacent enemies only if they're significantly weaker (quick win)."""
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b or actions[yi, xi] != 0:
                    continue

                for dir_idx in range(1, _NUM_ACTIONS):
                    dy, dx = _DIRECTION_OFFSETS[dir_idx]
                    ny, nx = (yi + dy) % height, (xi + dx) % width

                    n_owner = ownership[ny, nx]
                    n_strength = strength[ny, nx]

                    # Attack if neighbor is enemy and we're significantly stronger
                    if (n_owner != 0 and n_owner != agent_id_1b and 
                        strength[yi, xi] > n_strength * 1.5):
                        actions[yi, xi] = dir_idx
                        break

    def _expand_to_high_production(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        actions: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int
    ) -> None:
        """Prioritize expanding into high-production neutral cells."""
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b or actions[yi, xi] != 0:
                    continue

                best_dir = 0
                best_production = -1.0

                for dir_idx in range(1, _NUM_ACTIONS):
                    dy, dx = _DIRECTION_OFFSETS[dir_idx]
                    ny, nx = (yi + dy) % height, (xi + dx) % width

                    n_owner = ownership[ny, nx]
                    n_strength = strength[ny, nx]

                    # Only consider neutral cells with high production that we can take
                    if (n_owner == 0 and 
                        strength[yi, xi] > n_strength and
                        self.normalized_production[ny, nx] >= self.production_threshold):
                        
                        if self.normalized_production[ny, nx] > best_production:
                            best_production = self.normalized_production[ny, nx]
                            best_dir = dir_idx

                if best_dir != 0:
                    actions[yi, xi] = best_dir

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

    def _move_toward_frontier_with_production(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        actions: np.ndarray,
        is_frontier: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int,
        seed: Optional[int]
    ) -> None:
        """Move cells toward frontier, prioritizing high-production directions."""
        for yi in range(height):
            for xi in range(width):
                if ownership[yi, xi] != agent_id_1b or actions[yi, xi] != 0:
                    continue

                # Lower threshold: move if strong enough (more aggressive expansion)
                if strength[yi, xi] >= self.base_strength:
                    # Find best direction toward frontier with high production
                    best_dir = self._find_direction_to_frontier_with_production(
                        ownership, strength, yi, xi, is_frontier, agent_id_1b, height, width
                    )
                    if best_dir != 0:
                        actions[yi, xi] = best_dir

    def _find_direction_to_frontier_with_production(
        self,
        ownership: np.ndarray,
        strength: np.ndarray,
        yi: int,
        xi: int,
        is_frontier: np.ndarray,
        agent_id_1b: int,
        height: int,
        width: int
    ) -> int:
        """Find the best direction toward frontier that also has high production."""
        best_dir = 0
        best_score = -1.0

        for dir_idx in range(1, _NUM_ACTIONS):
            dy, dx = _DIRECTION_OFFSETS[dir_idx]
            ny, nx = (yi + dy) % height, (xi + dx) % width

            n_owner = ownership[ny, nx]
            n_strength = strength[ny, nx]

            # Only consider directions we can move to
            if n_owner == agent_id_1b or strength[yi, xi] <= n_strength:
                continue

            # Prioritize directions that lead toward frontier with high production
            score = 0.0
            if is_frontier[ny, nx]:
                score += 0.5  # Bonus for frontier cells
            score += self.normalized_production[ny, nx] * 0.5  # Production preference

            if score > best_score:
                best_score = score
                best_dir = dir_idx

        return best_dir
