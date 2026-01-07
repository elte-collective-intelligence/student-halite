import gymnasium as gym
from gymnasium import spaces
import json
import os
import random
from termcolor import colored
import numpy as np
from src.env.types import State, Observation, Action
from typing import Optional, Tuple, Dict, Any
from src.env.utils_env import update_game_state
from src.env.generator import Generator, UniformGenerator, OriginalGenerator
from functools import cached_property
from src.env.rewards import RewardFn, TerritoryRewardFn, StrengthRewardFn, ProductionRewardFn
import uuid
from src.env.constants import _MIN_PLAYERS, _MAX_PLAYERS

class Halite(gym.Env):

    def __init__(
        self,
        num_agents: int = 2,
        grid_size: Tuple[int, int] = (5, 5),
        generator: Optional[Generator] = None,
        reward_fn: Optional[RewardFn] = None,
    ):

        if num_agents < _MIN_PLAYERS or num_agents > _MAX_PLAYERS:
            raise ValueError(f"Number of agents must be between {_MIN_PLAYERS} and {_MAX_PLAYERS}")

        self._generator = generator or OriginalGenerator(grid_size=grid_size, num_agents=num_agents)
        # Use territory rewards by default
        self._reward_fn = reward_fn or TerritoryRewardFn()

        self.time_limit = int(np.sqrt(grid_size[0] * grid_size[1]) * 10)

        self.grid_size = grid_size

        self.num_agents = num_agents
        self.agent_ids = np.arange(self.num_agents)
        
        # Initialize state
        self.state: Optional[State] = None
        self._np_random = None

        super().__init__()

    def __repr__(self) -> str:
        return (
            f"Halite(\n"
            f"\tgrid_width={self.grid_size[1]!r},\n"
            f"\tgrid_height={self.grid_size[0]!r},\n"
            f"\tnum_agents={self.num_agents!r}, \n"
            ")"
        )
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Observation, np.ndarray, bool, bool, Dict[str, Any]]:
        """Perform an environment step.

        Args:
            action: Array containing the action to take.
                - 0 NOOP
                - 1 UP
                - 2 RIGHT
                - 3 DOWN
                - 4 LEFT

        Returns:
            observation: Observation object corresponding to the current state.
            reward: Reward for each agent.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode was truncated (always False for this env).
            info: Additional information dictionary.
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        action = action.astype(np.int32)

        new_state = update_game_state(self.state, action)
        observation_dict = {
            "grid": new_state.grid,
            "step_count": new_state.step_count,
            "action_mask": new_state.action_mask
        }

        reward = self._reward_fn(self.state, new_state, action)

        terminated = self._check_game_over(new_state)
        
        self.state = new_state

        return observation_dict, reward, terminated, False, {}

    def _check_game_over(self, state: State) -> bool:
        """
        Check if the game is over.
        The game is over if:
        - The time limit is reached
        - There is only one agent alive in multiple agents game
        - There is only one agent and it has all the cells

        Args:
            state: State object containing the current state of the environment.

        Returns:
            bool: True if the game is over, False otherwise.
        """

        time_limit_reached = state.step_count >= self.time_limit
        multiple_agents_condition = (np.sum(state.alive) == 1) & (self.num_agents > 1)
        single_agent_condition = (self.num_agents == 1) & (np.sum(state.grid[0]) == state.grid[0].shape[0] * state.grid[0].shape[1])

        return bool(time_limit_reached | multiple_agents_condition | single_agent_condition)

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        state = self._generator(seed=seed)
        self.state = state
        observation_dict = {
            "grid": state.grid,
            "step_count": state.step_count,
            "action_mask": state.action_mask
        }
        return observation_dict, {}

    @property
    def observation_space(self) -> spaces.Dict:
        """Returns the observation space."""
        return spaces.Dict({
            "grid": spaces.Box(
                low=0,
                high=self.num_agents,
                shape=(3, self.grid_size[0], self.grid_size[1]),
                dtype=np.int32
            ),
            "action_mask": spaces.Box(
                low=False,
                high=True,
                shape=(self.num_agents, self.grid_size[0], self.grid_size[1], len(Action)),
                dtype=bool
            ),
            "step_count": spaces.Box(
                low=0,
                high=self.time_limit,
                shape=(),
                dtype=np.int32
            )
        })

    @property
    def action_space(self) -> spaces.MultiDiscrete:
        """Returns the action space. 5 actions for each agent and each cell in the grid:
        [0,1,2,3,4] -> [WAIT, UP, RIGHT, DOWN, LEFT].

        Since this is a multi-agent environment, the environment expects an array of actions.
        This array is of shape (num_agents, num_rows, num_cols).
        """
        return spaces.MultiDiscrete(
            nvec=[[[len(Action)] * self.grid_size[1]] * self.grid_size[0]] * self.num_agents
        )
    
    def save(self, states, actions, player_names=None, filepath=None):
        """
        Save game states and actions to a replay file.
        
        Args:
            states: List of State objects representing the game states
            actions: Array of actions taken between states
            player_names: Optional list of player names
            filepath: Optional custom filepath. If None, uses default halite_games/game_{uuid}.hlt
    
        """
        # Check inputs are not None
        if states is None or actions is None:
            raise ValueError(colored("[HALITE] ", "red") + "States and actions cannot be None")
        
        # Check actions length matches states length
        if len(actions) != len(states) - 1:
            raise ValueError(colored("[HALITE] ", "red") + 
                            f"Number of actions ({len(actions)}) should be one less than number of states ({len(states)})")
        
        # Determine filename
        if filepath is None:
            # Default behavior: use halite_games directory
            os.makedirs("halite_games", exist_ok=True)
            unique_id = uuid.uuid4().hex[:8]
            filename = f"halite_games/game_{unique_id}.hlt"
        else:
            # Use provided filepath
            filename = filepath
            # Create directory if it doesn't exist
            dirname = os.path.dirname(filename)
            if dirname:  # Only create directory if there's a directory component
                os.makedirs(dirname, exist_ok=True)
        
        # Convert inputs to numpy arrays
        first_state_grid = np.array(states[0].grid)
        height, width = first_state_grid.shape[1:]
        num_players = self.num_agents
        
        # Pre-compute productions
        productions = np.array(states[0].grid[2]).tolist()
        
        if player_names is None:
            player_names = [f"Player {i+1}" for i in range(num_players)]
        
        # Process frames
        def process_frame(state_grid):
            owner = state_grid[0]
            strength = state_grid[1]
            return np.stack([owner.astype(int), strength.astype(int)], axis=-1)
        
        # Process all states
        frames = np.array([state.grid for state in states])
        processed_frames = np.array([process_frame(frame) for frame in frames])
        frames_list = processed_frames.tolist()
        
        # Process moves
        def process_move(state_grid, action):
            owner = state_grid[0]
            # Create a mask for each player
            player_masks = [owner == (i+1) for i in range(num_players)]
            # Apply masks to actions
            move_grid = np.zeros_like(owner, dtype=int)
            for i in range(num_players):
                move_grid = np.where(player_masks[i], action[i], move_grid)
            return move_grid
        
        # Convert actions to numpy array
        actions_array = np.array(actions)
        state_grids = np.array([state.grid for state in states[:-1]])
        
        # Process moves
        moves = np.array([process_move(state_grids[i], actions_array[i]) for i in range(len(state_grids))])
        moves_list = moves.astype(int).tolist()
        
        # Create output dictionary
        output = {
            "version": 11,
            "width": width,
            "height": height,
            "num_players": num_players,
            "num_frames": len(frames_list),
            "player_names": player_names,
            "productions": productions,
            "frames": frames_list,
            "moves": moves_list,
        }
        
        # Save to file
        with open(filename, "w") as f:
            json.dump(output, f)
        
        halite_prefix = colored("[HALITE]", "blue")
        success_msg = colored(f" Game successfully saved to {filename}", "white")
        print(f"\n{halite_prefix} {success_msg}\n")

    def get_statistics(self, states: list[State], actions: list[np.ndarray]) -> Dict[str, Any]:
        """
        Computes game statistics over a trajectory of States and player moves.
        """

        if states is None or len(states) == 0:
            raise ValueError("States list cannot be empty.")

        num_players = self.num_agents
        H, W = states[0].grid.shape[1:]
        T = len(states) - 1  # number of steps (actions)

        territory_share = np.zeros((num_players, T + 1), dtype=np.float32)
        production_per_turn = np.zeros((num_players, T + 1), dtype=np.float32)
        frontier_length = np.zeros((num_players, T + 1), dtype=np.float32)
        relative_strength = np.zeros((num_players, T + 1), dtype=np.float32)
        cap_losses = np.zeros((num_players, T + 1), dtype=np.float32)           # per-step cumulative
        player_damage_dealt = np.zeros((num_players, T + 1), dtype=np.float32)  # per-step cumulative
        strength_traded_per_turn = np.zeros((num_players, T), dtype=np.float32)  # per-turn
        territory_gained_per_turn = np.zeros((num_players, T), dtype=np.float32)  # per-turn
        successful_captures_per_turn = np.zeros((num_players, T), dtype=np.float32)  # per-turn: captures from other players (not neutral)

        cumulative_cap_losses = np.zeros(num_players, dtype=np.float32)
        cumulative_player_damage = np.zeros(num_players, dtype=np.float32)
        
        # Track maximum values per player
        max_territory = np.zeros(num_players, dtype=np.float32)
        max_strength = np.zeros(num_players, dtype=np.float32)
        max_production = np.zeros(num_players, dtype=np.float32)

        total_cells = H * W
        production_map = states[0].grid[2]
        total_production = np.sum(production_map)

        # Frontier helper
        def get_frontier(owner_map, player):
            mask = (owner_map == player)
            frontier = 0
            for r in range(H):
                for c in range(W):
                    if not mask[r, c]:
                        continue
                    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
                        rr = (r + dr) % H
                        cc = (c + dc) % W
                        if owner_map[rr, cc] != player:
                            frontier += 1
                            break
            return frontier

        # -----------------------------
        # First pass: per-state metrics
        # -----------------------------
        for t, state in enumerate(states):
            owner = state.grid[0]
            strength = state.grid[1]
            total_strength = np.sum(strength)

            for p in range(num_players):
                player_id = p + 1
                # Territory
                territory_count = np.sum(owner == player_id)
                territory_share[p, t] = territory_count / total_cells
                max_territory[p] = max(max_territory[p], territory_count)
                
                # Production
                prod_count = np.sum(production_map[owner == player_id])
                production_per_turn[p, t] = prod_count / total_production
                max_production[p] = max(max_production[p], prod_count)
                
                # Strength
                strength_count = np.sum(strength[owner == player_id])
                max_strength[p] = max(max_strength[p], strength_count)
                
                # Frontier
                frontier_length[p, t] = get_frontier(owner, player_id)
                # Relative strength
                relative_strength[p, t] = strength_count / max(total_strength, 1e-6)

            # -----------------------------
            # Cap losses using moves
            # -----------------------------
            if t < len(actions):
                moves = actions[t]  # shape: (num_players, H, W)
                new_strength_map = np.zeros((num_players, H, W), dtype=np.float32)

                for p in range(num_players):
                    player_id = p + 1
                    step_cap_loss = 0  # cap losses this step
                    for y in range(H):
                        for x in range(W):
                            if owner[y, x] != player_id:
                                continue
                            move = moves[p, y, x]
                            prod = production_map[y, x] if move == 0 else 0
                            new_y, new_x = y, x
                            if move == 1:  # NORTH
                                new_y = (y - 1) % H
                            elif move == 2:  # EAST
                                new_x = (x + 1) % W
                            elif move == 3:  # SOUTH
                                new_y = (y + 1) % H
                            elif move == 4:  # WEST
                                new_x = (x - 1) % W
                            total = new_strength_map[p, new_y, new_x] + strength[y, x] + prod
                            if total > 255:
                                step_cap_loss += total - 255
                                new_strength_map[p, new_y, new_x] = 255
                            else:
                                new_strength_map[p, new_y, new_x] = total

                    cumulative_cap_losses[p] += step_cap_loss
                    cap_losses[p, t] = cumulative_cap_losses[p]

        # -----------------------------
        # Engagement efficiency & player damage
        # -----------------------------
        for t in range(T):
            owner_t = states[t].grid[0]
            strength_t = states[t].grid[1]
            owner_next = states[t + 1].grid[0]
            strength_next = states[t + 1].grid[1]

            for p in range(num_players):
                player_id = p + 1
                before = np.sum(strength_t[owner_t == player_id])
                after = np.sum(strength_next[owner_next == player_id])
                lost = max(before - after, 0)
                strength_traded_per_turn[p, t] = lost

                territory_before = np.sum(owner_t == player_id)
                territory_after = np.sum(owner_next == player_id)
                gained = max(territory_after - territory_before, 0)
                territory_gained_per_turn[p, t] = gained
                
                # Track successful captures: cells that were owned by another player (not neutral) and are now owned by this player
                prev_owned_by_others = (owner_t != player_id) & (owner_t > 0)  # Exclude neutral (0) and current player
                now_owned_by_player = (owner_next == player_id)
                successful_captures = np.sum(prev_owned_by_others & now_owned_by_player)
                successful_captures_per_turn[p, t] = successful_captures

                # Approximate player damage dealt (overkill)
                damage_done = max(0, after - before)
                cumulative_player_damage[p] += damage_done
                player_damage_dealt[p, t+1] = cumulative_player_damage[p]

        # Return 3D array: (num_players, T, 2) where last dimension is [strength_traded, territory_gained] per turn
        engagement_efficiency = np.stack([strength_traded_per_turn, territory_gained_per_turn], axis=2)

        return {
            "territory_share": territory_share,
            "production_per_turn": production_per_turn,
            "frontier_length": frontier_length,
            "relative_strength": relative_strength,
            "engagement_efficiency": engagement_efficiency,
            "cap_losses": cap_losses,
            "player_damage_dealt": player_damage_dealt,
            "max_territory": max_territory,
            "max_strength": max_strength,
            "max_production": max_production,
            "successful_captures_per_turn": successful_captures_per_turn,
        }
