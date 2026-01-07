import numpy as np

_MIN_PLAYERS = 2
_MAX_PLAYERS = 6

_MIN_GRID_SIZE = 20
_MAX_GRID_SIZE = 50

_NEW_CELL_STRENGTH = 0
_MAX_STRENGTH = 255
_NUM_ACTIONS = 5

_DIRECTION_OFFSETS = np.array([
    [0, 0],    # STILL
    [-1, 0],   # NORTH
    [0, 1],    # EAST
    [1, 0],    # SOUTH
    [0, -1]    # WEST
])

_DAMAGE_KERNEL = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.float32)