"""
Grid search over RuleBasedBot parameters and visualize results as a heatmap.
Uses Hydra for configuration management.
"""

import numpy as np
from typing import List, Tuple, Dict
from joblib import Parallel, delayed
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.bots.rule_based import RuleBasedBot
from src.env.env import Halite
from src.viz.grid_search.grid_search_heatmap import create_grid_search_heatmap


def _evaluate_bot_configuration(
    test_base_strength: int,
    test_production_enemy_preference: float,
    grid_size: Tuple[int, int],
    num_agents: int,
    base_strength_range: List[int],
    production_enemy_preference_range: List[float],
    episodes: int,
    base_seed: int,
    task_idx: int
) -> float:
    """
    Evaluate a RuleBasedBot configuration against randomly configured opponents.
    A new bot instance is created for each game, and opponents get random, different
    configurations based on the episode seed.
    
    Parameters
    ----------
    test_base_strength : int
        base_strength for the test bot (agent_id=0)
    test_production_enemy_preference : float
        production_enemy_preference for the test bot (agent_id=0) (0 = always enemies, 1 = always production)
    grid_size : Tuple[int, int]
        Fixed grid size for all games
    num_agents : int
        Fixed number of agents (1 test bot + opponents)
    base_strength_range : List[int]
        Range of base_strength values to sample opponents from
    production_enemy_preference_range : List[float]
        Range of production_enemy_preference values to sample opponents from (0 = always enemies, 1 = always production)
    episodes : int
        Number of episodes to run
    base_seed : int
        Base random seed
    task_idx : int
        Task index for generating unique seeds
        
    Returns
    -------
    float
        Win rate (0.0 to 1.0) for the test bot
    """
    wins = 0
    num_opponents = num_agents - 1  # Number of opponent bots
    
    for ep in range(episodes):
        # Generate unique seed for this episode from base seed
        # Use task_idx and episode index to ensure uniqueness
        ep_seed = base_seed + task_idx * 10000 + ep * 100
        
        # Create RNG for this episode to generate opponent configurations
        rng = np.random.RandomState(ep_seed)
        
        # Create environment with fixed grid size
        env = Halite(grid_size=grid_size, num_agents=num_agents)
        
        # Reset with unique seed
        observation, info = env.reset(seed=ep_seed)
        state = env.state
        
        # Create new test bot instance (agent 0) for this game
        test_bot = RuleBasedBot(
            agent_id=0,
            name=f"TestBot_bs{test_base_strength}_pep{test_production_enemy_preference}",
            base_strength=test_base_strength,
            production_enemy_preference=test_production_enemy_preference
        )
        
        # Generate random, different opponent configurations for this episode
        # Sample without replacement to ensure all opponents are different
        opponent_configs = []
        used_configs = set()
        
        for a_id in range(1, num_agents):
            # Try to find a unique configuration
            max_attempts = 100
            for attempt in range(max_attempts):
                opp_base_strength = rng.choice(base_strength_range)
                opp_production_enemy_preference = rng.choice(production_enemy_preference_range)
                config_tuple = (opp_base_strength, opp_production_enemy_preference)
                
                if config_tuple not in used_configs:
                    used_configs.add(config_tuple)
                    opponent_configs.append({
                        'base_strength': opp_base_strength,
                        'production_enemy_preference': opp_production_enemy_preference
                    })
                    break
            else:
                # If we couldn't find a unique config, just use a random one
                # (this should rarely happen if ranges are large enough)
                opp_base_strength = rng.choice(base_strength_range)
                opp_production_enemy_preference = rng.choice(production_enemy_preference_range)
                opponent_configs.append({
                    'base_strength': opp_base_strength,
                    'production_enemy_preference': opp_production_enemy_preference
                })
        
        # Create new opponent bot instances with random configurations
        agents = {0: test_bot}
        for a_id in range(1, num_agents):
            opp_config = opponent_configs[a_id - 1]
            
            agents[a_id] = RuleBasedBot(
                agent_id=a_id,
                name=f"OpponentBot_bs{opp_config['base_strength']}_pep{opp_config['production_enemy_preference']}",
                base_strength=opp_config['base_strength'],
                production_enemy_preference=opp_config['production_enemy_preference']
            )
        
        # Track death times
        death_times = {a: None for a in range(num_agents)}
        
        # Run episode
        terminated = False
        while not terminated:
            # Register death times
            for a in range(num_agents):
                if death_times[a] is None and not state.alive[a]:
                    death_times[a] = state.step_count - 1
            
            # Compute actions
            # Use a per-step seed derived from episode seed and step count for deterministic behavior
            step_seed = ep_seed + state.step_count if ep_seed is not None else None
            step_actions = []
            for a in range(num_agents):
                step_actions.append(agents[a](observation["grid"], seed=step_seed))
            action = np.stack(step_actions)
            
            # Step
            observation, reward, terminated, truncated, info = env.step(action)
            state = env.state
        
        # Final death times
        max_step = state.step_count
        for a in range(num_agents):
            if death_times[a] is None and not state.alive[a]:
                death_times[a] = max_step
        
        # Determine winner (survived longest, then most territory)
        ownership_grid = state.grid[0]
        territory_counts = {
            a: np.sum(ownership_grid == (a + 1))
            for a in range(num_agents)
        }
        
        stats = []
        for a in range(num_agents):
            dt = death_times[a] if death_times[a] is not None else max_step
            stats.append((a, dt, territory_counts[a]))
        
        ranked = sorted(stats, key=lambda x: (-x[1], -x[2]))
        winner = ranked[0][0]
        
        if winner == 0:
            wins += 1
    
    return wins / episodes


def _evaluate_single_combination(
    i: int,
    j: int,
    base_strength: int,
    production_enemy_preference: float,
    grid_size: Tuple[int, int],
    num_agents: int,
    base_strength_range: List[int],
    production_enemy_preference_range: List[float],
    episodes: int,
    base_seed: int,
    task_idx: int
) -> Tuple[int, int, float]:
    """
    Evaluate a single parameter combination. Used for parallel execution.
    
    Returns
    -------
    Tuple[int, int, float]
        (i, j, win_rate) - matrix indices and win rate
    """
    win_rate = _evaluate_bot_configuration(
        test_base_strength=base_strength,
        test_production_enemy_preference=production_enemy_preference,
        grid_size=grid_size,
        num_agents=num_agents,
        base_strength_range=base_strength_range,
        production_enemy_preference_range=production_enemy_preference_range,
        episodes=episodes,
        base_seed=base_seed,
        task_idx=task_idx
    )
    return i, j, win_rate


@hydra.main(version_base=None, config_path="../../configs/eval", config_name="grid_search_rule_based_bot")
def main(cfg: DictConfig):
    """
    Perform a grid search over base_strength and production_enemy_preference parameters for RuleBasedBot2.
    Each configuration is evaluated against fixed opponents with fixed grid size.
    """
    print("Grid Search Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("\n" + "="*50 + "\n")
    
    # Extract configuration and convert OmegaConf ListConfig to regular Python lists
    base_strength_range = OmegaConf.to_container(cfg.grid_search.base_strength_range, resolve=True)
    production_enemy_preference_range = OmegaConf.to_container(cfg.grid_search.production_enemy_preference_range, resolve=True)
    episodes = cfg.grid_search.episodes
    base_seed = cfg.grid_search.seed
    
    grid_size = tuple(cfg.evaluation.grid_size)
    num_agents = cfg.evaluation.num_agents
    num_opponents = num_agents - 1  # 1 test bot + opponents
    
    # Validate that we have enough unique configurations possible
    max_unique_configs = len(base_strength_range) * len(production_enemy_preference_range)
    if num_opponents > max_unique_configs:
        raise ValueError(
            f"Not enough unique configurations possible: need {num_opponents} opponents but "
            f"only {max_unique_configs} unique combinations available from "
            f"base_strength_range={base_strength_range} and production_enemy_preference_range={production_enemy_preference_range}"
        )
    
    # Get Hydra output directory (where config files are saved)
    # This ensures the image is saved alongside the Hydra config
    hydra_cfg = HydraConfig.get()
    viz_dir = Path(hydra_cfg.runtime.output_dir)
    
    print(f"Starting grid search over {len(base_strength_range)} x {len(production_enemy_preference_range)} = "
          f"{len(base_strength_range) * len(production_enemy_preference_range)} parameter combinations...")
    print(f"Each combination will be evaluated over {episodes} episodes.")
    print(f"Grid size: {grid_size}")
    print(f"Number of agents: {num_agents} (1 test bot + {num_opponents} opponents)")
    print(f"Opponents will have random, different configurations sampled from:")
    print(f"  base_strength_range: {base_strength_range}")
    print(f"  production_enemy_preference_range: {production_enemy_preference_range}")
    print(f"Running experiments in parallel...")
    
    # Initialize result matrix
    win_rate_matrix = np.zeros((len(base_strength_range), len(production_enemy_preference_range)))
    
    # Prepare all parameter combinations for parallel execution
    tasks = []
    task_idx = 0
    for i, base_strength in enumerate(base_strength_range):
        for j, production_enemy_preference in enumerate(production_enemy_preference_range):
            tasks.append((
                i, j, base_strength, production_enemy_preference, task_idx
            ))
            task_idx += 1
    
    # Run all evaluations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(_evaluate_single_combination)(
            i=i,
            j=j,
            base_strength=base_strength,
            production_enemy_preference=production_enemy_preference,
            grid_size=grid_size,
            num_agents=num_agents,
            base_strength_range=base_strength_range,
            production_enemy_preference_range=production_enemy_preference_range,
            episodes=episodes,
            base_seed=base_seed,
            task_idx=task_idx
        )
        for i, j, base_strength, production_enemy_preference, task_idx in tasks
    )
    
    # Fill in the result matrix
    for i, j, win_rate in results:
        win_rate_matrix[i, j] = win_rate
        base_strength = base_strength_range[i]
        production_enemy_preference = production_enemy_preference_range[j]
        print(f"base_strength={base_strength}, production_enemy_preference={production_enemy_preference}: win_rate={win_rate:.3f}")
    
    # Create heatmap
    heatmap_path = create_grid_search_heatmap(
        win_rate_matrix,
        base_strength_range,
        production_enemy_preference_range,
        episodes,
        str(viz_dir)
    )
    
    print(f"\nGrid search complete!")
    print(f"Best win rate: {np.max(win_rate_matrix):.3f}")
    print(f"Mean win rate: {np.mean(win_rate_matrix):.3f}")
    print(f"Heatmap saved to: {heatmap_path}")


if __name__ == "__main__":
    main()
