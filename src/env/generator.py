import abc
from src.env.types import State
from typing import Tuple, Callable, Optional
import numpy as np
from src.env.utils_env import _get_action_mask
import math

class Generator(abc.ABC):
    """
    Base class for generators for the Halite environment.
    """

    def __init__(self, grid_size: Tuple[int, int], num_agents: int) -> None:
        """
        Initializes the generator.
        """
        self.grid_size = grid_size
        self.num_agents = num_agents

    @abc.abstractmethod
    def __call__(self, seed: Optional[int] = None) -> State:
        """
        Generates an `Halite` environment.
        Returns:
            An `Halite` state.
        """

class UniformGenerator(Generator):
    """
    A generator that generates a uniform map with homogeneous production and zero strength.
    """

    def __init__(self, grid_size: Tuple[int, int], num_agents: int):
        super().__init__(grid_size, num_agents)
        self.height, self.width = grid_size
        self.num_players = num_agents
        
        # Precompute both possible dimension configurations
        self.dims_h = self._compute_dimensions(True)
        self.dims_v = self._compute_dimensions(False)
        
        # Use the larger dimensions to create static arrays
        self.final_height = max(self.dims_h[5], self.dims_v[5])
        self.final_width = max(self.dims_h[4], self.dims_v[4])

    def _compute_dimensions(self, prefer_horizontal: bool) -> Tuple[int, int, int, int, int, int]:
        """Compute concrete dimensions that work with the player arrangement."""
        height, width = self.height, self.width
        num_players = self.num_players
        
        # Calculate dw and dh based on preference
        if prefer_horizontal:
            dh = int(math.sqrt(num_players))
            while num_players % dh != 0:
                dh -= 1
            dw = num_players // dh
        else:
            dw = int(math.sqrt(num_players))
            while num_players % dw != 0:
                dw -= 1
            dh = num_players // dw
        
        # Calculate chunk sizes
        cw = width // dw
        ch = height // dh
        
        # Adjust chunk sizes to be divisible by num_players if needed
        if prefer_horizontal:
            ch = max(1, (ch // num_players) * num_players)
        else:
            cw = max(1, (cw // num_players) * num_players)
        
        # Calculate final grid size that fits the chunks
        final_width = cw * dw
        final_height = ch * dh
        
        return (dw, dh, cw, ch, final_width, final_height)

    def __call__(self, seed: Optional[int] = None) -> State:
        """
        Generates a uniform state with:
        - Random horizontal/vertical preference
        - Random player placement order
        """
        rng = np.random.default_rng(seed)
        prefer_horizontal = rng.random() < 0.5
        
        # Create arrays with static size
        production = np.ones((self.final_height, self.final_width), dtype=np.int32)
        strength = np.zeros((self.final_height, self.final_width), dtype=np.int32)
        owner = np.zeros((self.final_height, self.final_width), dtype=np.int32)
        
        def create_grid(dims):
            dw, dh, cw, ch, final_width, final_height = dims
            
            # Generate all possible player positions
            a_indices = np.arange(dh)
            b_indices = np.arange(dw)
            a_all = np.repeat(a_indices, dw)
            b_all = np.tile(b_indices, dh)
            
            # Create random permutation for player order
            player_order = rng.permutation(dw*dh)
            
            # Place players in random order
            owner_placed = owner.copy()
            for i in range(dw*dh):
                idx = player_order[i]
                a = a_all[idx]
                b = b_all[idx]
                center_y = a * ch + ch // 2
                center_x = b * cw + cw // 2
                owner_placed[center_y, center_x] = i + 1  # Players are 1-indexed
            
            # Create mask for valid cells
            mask = np.zeros((self.final_height, self.final_width), dtype=bool)
            mask[:final_height, :final_width] = True
            
            # Apply mask to all layers
            owner_masked = np.where(mask, owner_placed, 0)
            strength_masked = np.where(mask, strength, 0)
            production_masked = np.where(mask, production, 0)
            
            return np.stack([owner_masked, strength_masked, production_masked], axis=0)
        
        if prefer_horizontal:
            grid = create_grid(self.dims_h)
        else:
            grid = create_grid(self.dims_v)
        
        action_mask = _get_action_mask(grid, self.num_players)
        alive = np.ones((self.num_players,), dtype=bool)
        
        return State(
            grid=grid,
            step_count=0,
            alive=alive,
            action_mask=action_mask
        )


class OriginalGenerator(Generator):
    """
    A generator that replicates the behavior of the OriginalMap from the C++ code,
    creating a map with procedurally generated production and strength values.
    """

    def __init__(self, grid_size: Tuple[int, int], num_agents: int) -> None:
        super().__init__(grid_size, num_agents)
        self.BASE_PROD = 6  # Base value for production
        self.BASE_STR = 150  # Base value for strength

    def __call__(self, seed: Optional[int] = None) -> State:

        height, width = self.grid_size
        num_players = self.num_agents
        
        rng = np.random.default_rng(seed)
        
        # Decide whether to prefer horizontal or vertical division
        prefer_horizontal = rng.random() < 0.5
        
        # Calculate dimensions
        sqrt_n = np.sqrt(num_players)
        base = int(np.floor(sqrt_n))
        
        def find_divisor(d):
            while num_players % d != 0:
                d -= 1
            return d
        
        # Calculate dimensions based on preferred orientation
        if prefer_horizontal:
            dw = find_divisor(base)
            dh = num_players // dw
        else:
            dh = find_divisor(base)
            dw = num_players // dh
        
        dw, dh = int(dw), int(dh)
        
        # Calculate chunk sizes
        cw = width // dw
        ch = height // dh
        
        # Adjust map size to fit chunks exactly
        width = cw * dw
        height = ch * dh
        
        def create_region(w: int, h: int, levels: int = 4) -> np.ndarray:
            CHUNK_SIZE = 4

            def subdivide(w, h):
                # Compute base chunk sizes
                cw = w // CHUNK_SIZE
                ch = h // CHUNK_SIZE
                difW = w - CHUNK_SIZE * cw
                difH = h - CHUNK_SIZE * ch

                # For each subregion, compute its size
                a_indices = np.arange(CHUNK_SIZE)
                b_indices = np.arange(CHUNK_SIZE)
                tch = ch + (a_indices < difH).astype(np.int32)
                tcw = cw + (b_indices < difW).astype(np.int32)
                valid_mask = (tch[:, None] > 0) & (tcw[None, :] > 0)

                # Generate random factors for each subregion
                uniform_vals = rng.random(CHUNK_SIZE * CHUNK_SIZE)
                factors = np.where(valid_mask, np.power(uniform_vals.reshape(CHUNK_SIZE, CHUNK_SIZE), 1.5), 0.0)

                return factors, tch, tcw, valid_mask

            def blur(factors, OWN_WEIGHT=0.75):
                mh = np.roll(factors, 1, axis=0)
                ph = np.roll(factors, -1, axis=0)
                mw = np.roll(factors, 1, axis=1)
                pw = np.roll(factors, -1, axis=1)
                return (factors * OWN_WEIGHT +
                        (mh + ph + mw + pw) * (1 - OWN_WEIGHT) / 4)

            def build(w, h, level):
                # Base case: 1x1 region or max depth
                if (w == 1 and h == 1) or (level == 0):
                    factor = np.power(rng.random(), 1.5)
                    return np.full((h, w), factor)

                # Subdivide
                factors, tch, tcw, valid_mask = subdivide(w, h)
                # Blur at this level
                factors = blur(factors)

                # For each subregion, recursively build its subgrid
                grids = []
                for a in range(CHUNK_SIZE):
                    row = []
                    for b in range(CHUNK_SIZE):
                        sub_h = tch[a]
                        sub_w = tcw[b]
                        if (sub_h > 0) and (sub_w > 0):
                            subgrid = build(sub_w, sub_h, level - 1)
                            # Multiply by the factor for this subregion
                            row.append(subgrid * factors[a, b])
                        else:
                            row.append(np.zeros((sub_h, sub_w)))
                    if row:
                        row_grid = np.concatenate(row, axis=1) if len(row) > 1 else row[0]
                        grids.append(row_grid)
                if grids:
                    full_grid = np.concatenate(grids, axis=0) if len(grids) > 1 else grids[0]
                else:
                    full_grid = np.zeros((h, w))
                return full_grid

            return build(w, h, levels)
        
        # Create production and strength regions
        prod_factors = create_region(cw, ch)
        str_factors = create_region(cw, ch)
        
        # Tile the regions for each player
        def tile_region(factors, dw, dh, cw, ch):
            a = np.arange(dh)
            b = np.arange(dw)
            c = np.arange(ch)
            d = np.arange(cw)
            
            # Calculate reflection conditions (vectorized)
            v_reflect = (dh % 2 == 0) & (a % 2 != 0)
            h_reflect = (dw % 2 == 0) & (b % 2 != 0)
            
            # Create all combinations of indices
            a_idx, b_idx, c_idx, d_idx = np.meshgrid(a, b, c, d, indexing='ij')
            
            # Calculate global coordinates
            y = a_idx * ch + c_idx
            x = b_idx * cw + d_idx
            
            # Calculate reflected coordinates
            rc = np.where(v_reflect[a_idx], ch - c_idx - 1, c_idx)
            rd = np.where(h_reflect[b_idx], cw - d_idx - 1, d_idx)
            
            # Scale factors to chunk size
            scaled_c = (rc * factors.shape[0]) // ch
            scaled_d = (rd * factors.shape[1]) // cw
            
            # Gather all values at once
            values = factors[scaled_c, scaled_d]
            
            # Create output array and scatter values
            tiled = np.zeros((height, width))
            tiled[y, x] = values
            
            return tiled
        
        prod_tiled = tile_region(prod_factors, dw, dh, cw, ch)
        str_tiled = tile_region(str_factors, dw, dh, cw, ch)
        
        # Apply shift
        if num_players == 6:
            # No shift for 6 players
            pass
        elif prefer_horizontal:
            shift = (rng.integers(0, dw) * ch) % height
            prod_tiled = np.roll(prod_tiled, shift, axis=0)
            str_tiled = np.roll(str_tiled, shift, axis=0)
        else:
            shift = (rng.integers(0, dh) * cw) % width
            prod_tiled = np.roll(prod_tiled, shift, axis=1)
            str_tiled = np.roll(str_tiled, shift, axis=1)

        # Apply final blur
        OWN_WEIGHT = 0.66667
        num_blurs = int(2 * np.sqrt(width * height) / 10)

        prod_blurred = prod_tiled.copy()
        str_blurred = str_tiled.copy()
        
        for _ in range(num_blurs):
            # Create shifted versions for neighbors
            mh = np.roll(prod_blurred, 1, axis=0)  # minus height
            ph = np.roll(prod_blurred, -1, axis=0)  # plus height
            mw = np.roll(prod_blurred, 1, axis=1)   # minus width
            pw = np.roll(prod_blurred, -1, axis=1)  # plus width
            
            # Compute new production values (vectorized)
            prod_blurred = (prod_blurred * OWN_WEIGHT + 
                    (mh + ph + mw + pw) * (1 - OWN_WEIGHT) / 4)
            
            # Repeat for strength
            mh_str = np.roll(str_blurred, 1, axis=0)
            ph_str = np.roll(str_blurred, -1, axis=0)
            mw_str = np.roll(str_blurred, 1, axis=1)
            pw_str = np.roll(str_blurred, -1, axis=1)
            
            str_blurred = (str_blurred * OWN_WEIGHT + 
                        (mh_str + ph_str + mw_str + pw_str) * (1 - OWN_WEIGHT) / 4)

        # Normalize
        max_prod = np.max(prod_blurred)
        max_str = np.max(str_blurred)
        prod_normalized = prod_blurred / max_prod if max_prod > 0 else prod_blurred
        str_normalized = str_blurred / max_str if max_str > 0 else str_blurred
        
        # Scale to final values
        top_prod = self.BASE_PROD + rng.integers(0, 10)
        top_str = self.BASE_STR + rng.integers(0, 106)
        
        production = np.round(prod_normalized * top_prod).astype(np.int32)
        strength = np.round(str_normalized * top_str).astype(np.int32)
        
        def create_owner_grid(dh: int, dw: int, ch: int, cw: int, height: int, width: int) -> np.ndarray:
            # Create grid of chunk centers
            a = np.arange(dh)
            b = np.arange(dw)
            
            # Calculate center coordinates
            center_y = a * ch + ch // 2
            center_x = b * cw + cw // 2
            
            # Create meshgrid of all center positions
            y_coords, x_coords = np.meshgrid(center_y, center_x, indexing='ij')
            
            # Calculate owner IDs (a*dw + b + 1)
            owner_ids = (a[:, None] * dw + b[None, :] + 1).astype(np.int32)
            
            # Apply offsets based on player ID:
            # Player 2: move one square to the left (x -= 1)
            # Player 3: move one pixel up (y -= 1)
            # Player 4: move one square to the left AND one pixel up (x -= 1, y -= 1)
            x_offsets = np.where(owner_ids == 2, -1, np.where(owner_ids == 4, -1, 0))
            y_offsets = np.where(owner_ids == 3, -1, np.where(owner_ids == 4, -1, 0))
            
            # Apply offsets with torus wrapping
            x_coords = (x_coords + x_offsets) % width
            y_coords = (y_coords + y_offsets) % height
            
            # Create empty grid and scatter owner IDs
            owner = np.zeros((height, width), dtype=np.int32)
            owner[y_coords, x_coords] = owner_ids
            
            return owner
    
        # Create owner grid
        owner = create_owner_grid(dh, dw, ch, cw, height, width)
        
        # Ensure player cells have at least production 1
        production = np.where((owner > 0) & (production == 0), 1, production)
        
        # Stack the channels to create the grid
        grid = np.stack([owner, strength, production], axis=0)
        
        # Create action mask
        action_mask = _get_action_mask(grid, self.num_agents)
        
        # All agents are alive initially
        alive = np.ones((self.num_agents,), dtype=bool)
        
        return State(
            grid=grid,
            step_count=0,
            alive=alive,
            action_mask=action_mask
        )