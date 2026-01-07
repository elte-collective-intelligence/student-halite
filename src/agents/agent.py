from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class Agent(ABC):
    """Abstract base class for Halite agents."""
    
    def __init__(self, agent_id: int, name: str = "Agent"):
        self.agent_id = agent_id  # 0-based agent ID
        self.name = name
    
    @abstractmethod
    def __call__(self, observation: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Generate actions for the current observation.
        
        Args:
            observation: Environment observation (grid)
            seed: Optional random seed
            
        Returns:
            Action array of shape [height, width] with actions only for owned cells
        """
        pass