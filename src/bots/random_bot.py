import numpy as np
from typing import Optional

from src.agents.agent import Agent
from src.env.constants import _NUM_ACTIONS


class RandomBot(Agent):
    """Agent that takes random valid actions on owned cells."""

    def __init__(self, agent_id: int, name: str = "RandomBot"):
        super().__init__(agent_id, name)

    def __call__(self, observation: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Generate random actions only for owned cells (where ownership == agent_id + 1)."""
        ownership = observation[0]  # Ownership channel (1-based)
        height, width = ownership.shape

        # Create action mask (True where this agent owns cells)
        action_mask = ownership == self.agent_id + 1  # +1 because env uses 1-based IDs

        # Generate random actions for all cells
        rng = np.random.default_rng(seed)
        actions = rng.integers(0, _NUM_ACTIONS, size=(height, width))

        # Zero out actions for non-owned cells
        return np.where(action_mask, actions, 0)


