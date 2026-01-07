from enum import IntEnum
from typing import NamedTuple
import numpy as np
from dataclasses import dataclass

class Action(IntEnum): 
    """
    An enumeration of actions that are the possible actions that a player can take.
    WAIT: Do nothing.
    UP: Move up.
    RIGHT: Move right.
    DOWN: Move down.
    LEFT: Move left.
    """
    WAIT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4

class State(NamedTuple):
    """
    A class that represents the state of the game.
    grid: The grid of the game.
        - 3 channels: owner, strength, production 
    step_count: The number of steps in the episode.
    action_mask: The mask of the actions that are possible in the current state.
    """
    grid: np.ndarray  # (3, num_rows, num_cols) # 3 channels: owner, strength, production 
    step_count: int  # ()
    alive: np.ndarray # (num_agents,)
    action_mask: np.ndarray # (num_agents, num_rows, num_cols, num_actions)

class Observation(NamedTuple):
    """
    A class that represents the observation of the game.
    grid: The grid of the game.
        - 3 channels: owner, strength, production 
    step_count: The number of steps in the episode.
    action_mask: The mask of the actions that are possible in the current state.
    """
    grid: np.ndarray  # (3, num_rows, num_cols) # 3 channels: owner, strength, production 
    step_count: int  # ()
    action_mask: np.ndarray # (num_agents, num_rows, num_cols, num_actions)