import numpy as np
from typing import Optional

from src.agents.agent import Agent
from src.env.constants import _NUM_ACTIONS, _DIRECTION_OFFSETS

import numpy as np
from typing import Optional, Tuple

from src.agents.agent import Agent
from src.env.constants import _NUM_ACTIONS, _DIRECTION_OFFSETS

class RuleBasedV1(Agent):
    """Agent that either attacks adjacent enemies or moves randomly."""

    def __init__(self, agent_id: int, name: str = "RB_V1"):
        super().__init__(agent_id, name)

    def __call__(self, observation: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        ownership = observation[0]  # Ownership channel (1-based)
        strength = observation[1]  # Strength channel
        height, width = ownership.shape
        agent_id_1b = self.agent_id + 1  # 1-based agent id

        # Initialize all actions to WAIT (0)
        actions = np.zeros((height, width), dtype=np.int32)

        # Create coordinate grids
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # For each possible direction (skip WAIT=0)
        for dir_idx in range(1, _NUM_ACTIONS):
            dy, dx = _DIRECTION_OFFSETS[dir_idx]
            ny, nx = (y + dy) % height, (x + dx) % width

            # Get neighbor info
            neighbor_owned = ownership[ny, nx]
            neighbor_strength = strength[ny, nx]

            # Condition for attacking: neighbor is enemy and we're stronger
            attack_condition = (
                (ownership == agent_id_1b)  # Our cell
                & (neighbor_owned != agent_id_1b)  # Neighbor is enemy
                & (strength > neighbor_strength)  # We're stronger
            )

            # Set attack direction where conditions are met
            actions = np.where(attack_condition, dir_idx, actions)

        # For cells that didn't attack, move randomly if they're strong enough
        rng = np.random.default_rng(seed)
        random_dirs = rng.integers(
            1,  # Skip WAIT (0)
            _NUM_ACTIONS,
            size=(height, width),
        )

        move_condition = (
            (ownership == agent_id_1b)  # Our cell
            & (actions == 0)  # Didn't attack
            & (strength >= 40)  # Minimal strength to move
        )

        actions = np.where(move_condition, random_dirs, actions)

        return actions
