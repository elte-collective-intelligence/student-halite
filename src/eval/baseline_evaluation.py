import numpy as np
from typing import Tuple, Dict, Any, Optional
from joblib import Parallel, delayed
from hydra.utils import instantiate
import pickle
import copy
import tempfile
import os

from src.agents.agent import Agent
from src.bots.rule_based import RuleBasedBot
from src.bots.random_bot import RandomBot
from src.env.env import Halite


class AlgorithmAgent(Agent):
    """Wrapper agent that uses an algorithm instance for action selection."""
    
    def __init__(self, algorithm, agent_id: int = 0, name: str = "AlgorithmAgent", env=None):
        super().__init__(agent_id, name)
        self.algorithm = algorithm
        self.env = env
    
    def __call__(self, observation: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
        """Generate actions using the algorithm."""
        # observation is the grid (3, H, W)
        # Need to create action_mask if env is available
        action_mask = None
        if self.env is not None:
            # Get current state from env to compute action_mask
            if hasattr(self.env, 'state') and self.env.state is not None:
                action_mask = self.env.state.action_mask
        
        # If action_mask is still None, create a default one (all actions allowed)
        # This should not happen in normal operation, but provides a fallback
        if action_mask is None:
            H, W = observation.shape[1], observation.shape[2]
            num_agents = self.algorithm.env.num_agents if hasattr(self.algorithm, 'env') else 4
            action_mask = np.ones((num_agents, H, W, 5), dtype=bool)
        
        obs_dict = {
            "grid": observation,
            "step_count": 0,  # Not used for action selection
            "action_mask": action_mask
        }
        
        # Use algorithm's select_action method
        if hasattr(self.algorithm, 'select_action'):
            # For MAPPO, local_obs_dict is optional and will be generated automatically if not provided
            # Check if select_action signature requires local_obs_dict
            import inspect
            sig = inspect.signature(self.algorithm.select_action)
            params = list(sig.parameters.keys())
            
            if 'local_obs_dict' in params:
                # MAPPO - explicitly create local_obs_dict to match training behavior
                # This ensures the same observation format and action_mask handling as during training
                from src.training.training_util import create_local_observation_dict
                local_obs_dict = create_local_observation_dict(obs_dict, self.algorithm.env.num_agents)
                # Sample from distribution (matches training behavior)
                # This ensures proper action selection with masking
                # Pass seed for deterministic behavior
                action = self.algorithm.select_action(obs_dict, local_obs_dict=local_obs_dict, seed=seed)
            else:
                # Other algorithms (IQL, IPPO, CQL, etc.) - match training behavior
                # This ensures consistent action selection with proper masking and sampling
                # Pass seed for deterministic behavior
                action = self.algorithm.select_action(obs_dict, seed=seed)
            
            # Return action for this agent only (agent_id)
            # Handle both single action array and tuple (actions, log_probs, values)
            if isinstance(action, tuple):
                action = action[0]  # Extract actions from tuple
            return action[self.agent_id]
        else:
            raise ValueError("Algorithm does not have select_action method")


def create_agent_from_algorithm(algorithm, agent_id: int = 0, name: str = "TrainedAgent", env=None) -> AlgorithmAgent:
    """Create an agent wrapper from an algorithm instance.
    
    Args:
        algorithm: Algorithm instance (CQL, IQL, IPPO, MAPPO, etc.)
        agent_id: Agent ID (default 0)
        name: Agent name
        env: Environment instance (for action_mask computation)
        
    Returns:
        AlgorithmAgent instance
    """
    return AlgorithmAgent(algorithm, agent_id=agent_id, name=name, env=env)


def _run_single_episode_with_stats(
    test_agent_cfg: Dict[str, Any], 
    seed: int, 
    grid_size: Tuple[int, int],
    num_agents: int,
    opponent_bot_class: type,
    algorithm_instance: Optional[Any] = None,
    algorithm_path: Optional[str] = None,
    algorithm_config: Optional[Dict[str, Any]] = None
):
    """
    Runs ONE episode and returns:
        - win: 1 if test_agent (agent_id=0) wins, 0 otherwise
        - position: final position (1-indexed)
        - statistics dictionary from Halite.get_statistics
        - agent_names: list where agent_names[i] is the name of agent with id i
        - episode_length: number of steps
    
    Args:
        test_agent_cfg: Hydra config dict for agent (used if algorithm_instance is None)
        seed: Random seed
        grid_size: Grid dimensions
        num_agents: Number of agents
        opponent_bot_class: Bot class for opponents
        algorithm_instance: Optional algorithm instance (if provided, used instead of test_agent_cfg)
    """
    rng = np.random.RandomState(seed)

    # --- Build environment with fixed grid size and num agents ---
    env = Halite(grid_size=grid_size, num_agents=num_agents)
    
    # Instantiate test_agent in this parallel process
    if algorithm_path is not None and algorithm_config is not None:
        # Load algorithm from saved file
        from hydra.utils import get_class
        target_class = algorithm_config.get('_target_')
        if target_class is None:
            raise ValueError("algorithm_config must contain '_target_' field")
        
        algo_class = get_class(target_class)
        algo_kwargs = {k: v for k, v in algorithm_config.items() if k not in ['_target_', 'device']}
        algo_kwargs['env'] = env
        algo_kwargs['device'] = algorithm_config.get('device', 'cpu')
        
        # Create algorithm instance
        algo_instance = algo_class(**algo_kwargs)
        algo_instance.load(algorithm_path)
        
        # Set to eval mode
        import torch.nn as nn
        for attr_name in dir(algo_instance):
            attr = getattr(algo_instance, attr_name)
            if isinstance(attr, nn.Module):
                attr.eval()
        
        test_agent = create_agent_from_algorithm(algo_instance, agent_id=0, name="TrainedAgent", env=env)
    elif algorithm_instance is not None:
        # For single-threaded use (not recommended for parallel)
        test_agent = create_agent_from_algorithm(algorithm_instance, agent_id=0, name="TrainedAgent", env=env)
    else:
        test_agent = instantiate(test_agent_cfg)

    # Reset environment
    ep_seed = rng.randint(0, 1_000_000_000)
    observation, info = env.reset(seed=ep_seed)
    state = env.state

    # Agents dict - all opponents are the same bot type
    agents = {0: test_agent}
    for a_id in range(1, num_agents):
        agents[a_id] = opponent_bot_class(agent_id=a_id)

    # --- NEW: track agent names (by id order) ---
    agent_names = []
    for a_id in range(num_agents):
        if hasattr(agents[a_id], "name"):
            agent_names.append(agents[a_id].name)
        else:
            agent_names.append(f"Agent{a_id}")

    # Track death times
    death_times = {a: None for a in range(num_agents)}

    # Track all states and actions for statistics
    trajectory = [state]
    all_actions = []

    terminated = False

    # -----------------------
    #  Episode Loop
    # -----------------------
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
            # Update env state for action_mask if needed
            if hasattr(agents[a], 'env'):
                agents[a].env = env
            step_actions.append(agents[a](observation["grid"], seed=step_seed))
        action = np.stack(step_actions)
        all_actions.append(action)

        # Step
        observation, reward, terminated, truncated, info = env.step(action)
        state = env.state
        trajectory.append(state)

    # Final death times
    max_step = state.step_count
    for a in range(num_agents):
        if death_times[a] is None and not state.alive[a]:
            death_times[a] = max_step

    # --- Final territory ---
    ownership_grid = state.grid[0]
    territory_counts = {
        a: np.sum(ownership_grid == (a + 1))
        for a in range(num_agents)
    }

    # Rank players
    stats = []
    for a in range(num_agents):
        dt = death_times[a] if death_times[a] is not None else max_step
        stats.append((a, dt, territory_counts[a]))

    ranked = sorted(stats, key=lambda x: (-x[1], -x[2]))
    
    # Find position of agent 0 (1-indexed: 1st, 2nd, 3rd, etc.)
    position = next((i + 1 for i, (agent_id, _, _) in enumerate(ranked) if agent_id == 0), num_agents)

    # Compute statistics using Halite.get_statistics
    statistics = env.get_statistics(trajectory, actions=all_actions)

    # Return win/loss, position, stats, and agent names
    win = 1 if ranked[0][0] == 0 else 0
    return win, position, statistics, agent_names, max_step

from collections import defaultdict

def compute_opponent_win_rates(wins, agent_names_list):
    """
    Computes per-opponent win rates for agent 0.

    Parameters
    ----------
    wins : list[int]
        wins[i] = 1 if agent 0 won episode i else 0
    agent_names_list : list[list[str]]
        agent_names_list[i] = list of agent names for episode i, 
        aligned with agent IDs.

    Returns
    -------
    total_win_rate : float
    opponent_win_rates : dict[str, float]
        Example: {"RuleBot": 0.42, "EasyBot": 0.55}
    opponent_counts : dict[str, int]
        How many episodes each opponent appeared in.
    """
    assert len(wins) == len(agent_names_list)

    # Track how many times we faced each opponent
    opponent_appearances = defaultdict(int)

    # Track cumulative wins in episodes involving that opponent
    opponent_win_sum = defaultdict(int)

    win_rates = defaultdict(int)

    num_episodes = len(wins)

    for ep_idx in range(num_episodes):
        ep_win = wins[ep_idx]
        agent_names = agent_names_list[ep_idx]

        # Every opponent except agent 0
        for opp_name in set(agent_names[1:]):
            opponent_appearances[opp_name] += 1
            opponent_win_sum[opp_name] += ep_win  # agent 0 win counts toward every opponent in episode

    # Compute per-opponent win rates
    win_rates = {
        opp: opponent_win_sum[opp] / opponent_appearances[opp]
        for opp in opponent_appearances
    }

    win_rates["global"] = sum(wins) / num_episodes

    return win_rates


def compute_comprehensive_metrics(
    wins: list,
    positions: list,
    episode_lengths: list,
    stats_list: list,
    agent_names_list: list
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics from episode results.
    
    Parameters:
    -----------
    wins : list[int]
        List of win indicators (1 if agent 0 won, 0 otherwise)
    positions : list[int]
        List of final positions (1-indexed: 1st, 2nd, etc.)
    episode_lengths : list[int]
        List of episode lengths (time lasted)
    stats_list : list[dict]
        List of statistics dictionaries from get_statistics
    agent_names_list : list[list[str]]
        List of agent names per episode
    
    Returns:
    --------
    metrics : dict
        Comprehensive metrics dictionary
    """
    num_episodes = len(wins)
    if num_episodes == 0:
        return {}
    
    # 1. Win rates
    global_win_rate = sum(wins) / num_episodes
    
    # Compute per-opponent win rates
    opponent_win_rates = compute_opponent_win_rates(wins, agent_names_list)
    
    # 2. Average position
    avg_position = np.mean(positions)
    
    # 3. Time lasted (avg, max, min, std)
    time_lasted = {
        'avg': np.mean(episode_lengths),
        'max': np.max(episode_lengths),
        'min': np.min(episode_lengths),
        'std': np.std(episode_lengths)
    }
    
    # 4-6. Max territory, strength, production (avg + std)
    max_territories = [stats['max_territory'][0] for stats in stats_list]  # Agent 0
    max_strengths = [stats['max_strength'][0] for stats in stats_list]
    max_productions = [stats['max_production'][0] for stats in stats_list]
    
    max_territory = {
        'avg': np.mean(max_territories),
        'std': np.std(max_territories)
    }
    max_strength = {
        'avg': np.mean(max_strengths),
        'std': np.std(max_strengths)
    }
    max_production = {
        'avg': np.mean(max_productions),
        'std': np.std(max_productions)
    }
    
    # 7. Final cap losses (avg, max, min, std)
    final_cap_losses = [stats['cap_losses'][0, -1] for stats in stats_list]  # Agent 0, final step
    cap_losses_metrics = {
        'avg': np.mean(final_cap_losses),
        'max': np.max(final_cap_losses),
        'min': np.min(final_cap_losses),
        'std': np.std(final_cap_losses)
    }
    
    # 8. % of successful captures (avg, max, min, std)
    # Successful captures: cells captured from other players (not neutral)
    total_captures_per_episode = []
    for stats in stats_list:
        # Sum successful captures across all turns for agent 0
        total_captures = np.sum(stats['successful_captures_per_turn'][0])
        total_captures_per_episode.append(total_captures)
    
    # Calculate percentage: successful captures / total cells (or use a different denominator)
    # For now, we'll use the raw count, but you might want to normalize by total cells or total territory gained
    successful_captures_pct = {
        'avg': np.mean(total_captures_per_episode),
        'max': np.max(total_captures_per_episode),
        'min': np.min(total_captures_per_episode),
        'std': np.std(total_captures_per_episode)
    }
    
    # 9. Avg. engagement efficiency: strength traded vs. territory gained
    engagement_ratios = []
    for stats in stats_list:
        # Agent 0's engagement efficiency
        strength_traded = stats['engagement_efficiency'][0, :, 0]  # All turns
        territory_gained = stats['engagement_efficiency'][0, :, 1]  # All turns
        
        # Filter out turns where no strength was traded (to avoid division by zero)
        valid_turns = strength_traded > 0
        if np.any(valid_turns):
            ratios = territory_gained[valid_turns] / strength_traded[valid_turns]
            engagement_ratios.append(np.mean(ratios))
        else:
            engagement_ratios.append(0.0)
    
    avg_engagement_efficiency = np.mean(engagement_ratios) if engagement_ratios else 0.0
    
    return {
        'win_rate': {
            'global': global_win_rate,
            'by_opponent': opponent_win_rates
        },
        'avg_position': avg_position,
        'time_lasted': time_lasted,
        'max_territory': max_territory,
        'max_strength': max_strength,
        'max_production': max_production,
        'cap_losses': cap_losses_metrics,
        'successful_captures': successful_captures_pct,
        'engagement_efficiency': avg_engagement_efficiency
    }


def baseline_evaluation_with_stats(
    test_agent_cfg: Dict[str, Any] = None,
    episodes: int = 100,
    seed: int = 0,
    grid_size: Tuple[int, int] = (30, 30),
    num_agents: int = 4,
    opponent_bot_classes: list = None,
    algorithm_instance: Optional[Any] = None,
    algorithm_config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
):
    """
    Evaluates test_agent against different bot types and returns comprehensive metrics.
    
    Parameters:
    -----------
    test_agent_cfg : Dict[str, Any], optional
        Hydra configuration dict for the agent to evaluate (used if algorithm_instance is None)
    episodes : int
        Number of episodes per bot type
    seed : int
        Random seed
    grid_size : Tuple[int, int]
        Fixed grid size for all episodes
    num_agents : int
        Fixed number of agents for all episodes
    opponent_bot_classes : list
        List of bot classes to evaluate against. If None, uses [RandomBot, RuleBasedBot]
    algorithm_instance : Any, optional
        Algorithm instance to evaluate (if provided, used instead of test_agent_cfg)
    verbose : bool, default True
        If True, print detailed progress information. If False, only essential messages.
    
    Returns:
    --------
    metrics_by_bot : dict
        Dictionary mapping bot class names to comprehensive metrics dictionaries
    global_metrics : dict
        Overall metrics across all bot types
    """
    if algorithm_instance is None and test_agent_cfg is None:
        raise ValueError("Either test_agent_cfg or algorithm_instance must be provided")
    
    if opponent_bot_classes is None:
        opponent_bot_classes = [RandomBot, RuleBasedBot]
    
    # Handle algorithm instance by saving to temp file
    algorithm_path = None
    temp_file = None
    if algorithm_instance is not None:
        # Save algorithm to temporary file for parallel workers
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
        temp_file.close()
        algorithm_path = temp_file.name
        try:
            algorithm_instance.save(algorithm_path)
        except Exception as e:
            os.unlink(algorithm_path)
            raise RuntimeError(f"Failed to save algorithm for evaluation: {e}")
    
    if verbose:
        print(f"Running {episodes} episodes per bot type ({len(opponent_bot_classes)} bot types) in parallel...")
        print(f"Grid size: {grid_size}, Num agents: {num_agents}")

    metrics_by_bot = {}
    all_wins = []
    all_positions = []
    all_lengths = []
    all_stats = []
    all_agent_names = []

    try:
        for bot_class in opponent_bot_classes:
            bot_name = bot_class.__name__
            if verbose:
                print(f"Evaluating against {bot_name}...")
            
            rng = np.random.RandomState(seed)
            seeds = [rng.randint(0, 1_000_000_000) for _ in range(episodes)]

            # Run in parallel
            results = Parallel(n_jobs=-1)(
                delayed(_run_single_episode_with_stats)(
                    test_agent_cfg, s, grid_size, num_agents, bot_class, 
                    None, algorithm_path, algorithm_config
                )
                for s in seeds
            )

            # Separate fields
            wins, positions, stats_list, agent_names_list, lengths = zip(*results)
            
            # Compute metrics for this bot type
            bot_metrics = compute_comprehensive_metrics(
                list(wins), list(positions), list(lengths), 
                list(stats_list), list(agent_names_list)
            )
            metrics_by_bot[bot_name] = bot_metrics
            
            # Collect all data for global metrics
            all_wins.extend(wins)
            all_positions.extend(positions)
            all_lengths.extend(lengths)
            all_stats.extend(stats_list)
            all_agent_names.extend(agent_names_list)

        # Compute global metrics (once after all bots are evaluated)
        global_metrics = compute_comprehensive_metrics(
            all_wins, all_positions, all_lengths, all_stats, all_agent_names
        )
    finally:
        # Clean up temporary file
        if algorithm_path is not None and os.path.exists(algorithm_path):
            try:
                os.unlink(algorithm_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary algorithm file {algorithm_path}: {e}")

    return metrics_by_bot, global_metrics
