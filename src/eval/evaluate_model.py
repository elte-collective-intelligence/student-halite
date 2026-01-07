#!/usr/bin/env python3
"""Script to evaluate a model and plot the results.

Usage:
    python evaluate_model.py --model-folder <model_folder> [--checkpoint-name <checkpoint_name>] [--eval-config <eval_config_path>]
    
Example:
    python evaluate_model.py --model-folder models/mappo_B67C89
    python evaluate_model.py --model-folder models/mappo_B67C89 --checkpoint-name model.pt --eval-config configs/evaluation/baseline_evaluation.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import secrets
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate, get_class
from typing import Dict, Any, List, Tuple, Optional

# Register custom resolver for random hex generation
try:
    OmegaConf.register_new_resolver("random_hex", lambda length: secrets.token_hex((int(length) + 1) // 2).upper()[:int(length)])
except ValueError:
    pass

from src.eval.baseline_evaluation import baseline_evaluation_with_stats, _run_single_episode_with_stats, compute_comprehensive_metrics
from src.viz.eval.baseline_evaluation import save_halite_statistics_plot
from joblib import Parallel, delayed


def load_checkpoint_and_config(model_folder: str, checkpoint_name: Optional[str] = None) -> Tuple[Any, DictConfig, str]:
    """
    Load checkpoint and config from a models folder.
    
    Args:
        model_folder: Path to model folder (e.g., models/mappo_B67C89)
        checkpoint_name: Optional checkpoint filename (default: 'model.pt')
    
    Returns:
        Tuple of (algorithm_instance, config, device)
    """
    model_folder = Path(model_folder)
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    
    if not model_folder.is_dir():
        raise ValueError(f"Path is not a directory: {model_folder}")
    
    # Find config file
    config_path = model_folder / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Find checkpoint file
    if checkpoint_name is None:
        # Default to model.pt
        checkpoint_path = model_folder / 'model.pt'
        if not checkpoint_path.exists():
            # Try to find any .pt file in the folder
            pt_files = list(model_folder.glob('*.pt'))
            if len(pt_files) == 0:
                raise FileNotFoundError(f"No model file found in {model_folder}. Expected 'model.pt' or any .pt file.")
            elif len(pt_files) == 1:
                checkpoint_path = pt_files[0]
            else:
                # Multiple .pt files, prefer model.pt or latest checkpoint
                checkpoint_path = pt_files[0]  # Use first one found
                print(f"Warning: Multiple .pt files found in {model_folder}, using {checkpoint_path}")
    else:
        checkpoint_path = model_folder / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Determine device
    device = cfg.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Algorithm: {cfg.algo._target_}")
    
    # Create environment
    print("Creating environment...")
    env = instantiate(cfg.env)
    
    # Create algorithm
    print("Creating algorithm...")
    algo_kwargs = OmegaConf.to_container(cfg.algo, resolve=True)
    algo_kwargs.pop('use_local_observations', None)  # Remove if present
    algo_kwargs['env'] = env
    algo_kwargs['device'] = device
    
    target_class = algo_kwargs.pop('_target_')
    algo_class = get_class(target_class)
    algo = algo_class(**algo_kwargs)
    
    # Load checkpoint
    print(f"Loading checkpoint weights...")
    algo.load(str(checkpoint_path))
    print("✓ Checkpoint loaded successfully!")
    
    # Set to eval mode
    import torch.nn as nn
    for attr_name in dir(algo):
        attr = getattr(algo, attr_name)
        if isinstance(attr, nn.Module):
            attr.eval()
    
    return algo, cfg, device


def run_single_episode_for_replay(
    algorithm_path: str,
    algorithm_config: Dict[str, Any],
    seed: int,
    grid_size: Tuple[int, int],
    num_agents: int,
    opponent_bot_class: type
) -> Tuple[List[Any], List[np.ndarray], List[str]]:
    """
    Run a single episode and return states, actions, and agent names for saving replay.
    
    Returns:
        states: List of State objects
        actions: List of action arrays
        agent_names: List of agent names
    """
    from src.env.env import Halite
    from src.eval.baseline_evaluation import create_agent_from_algorithm
    from hydra.utils import get_class
    import torch.nn as nn
    
    rng = np.random.RandomState(seed)
    
    # Create environment
    env = Halite(grid_size=grid_size, num_agents=num_agents)
    
    # Load algorithm
    target_class = algorithm_config.get('_target_')
    algo_class = get_class(target_class)
    algo_kwargs = {k: v for k, v in algorithm_config.items() if k not in ['_target_', 'device']}
    algo_kwargs['env'] = env
    algo_kwargs['device'] = algorithm_config.get('device', 'cpu')
    
    algo_instance = algo_class(**algo_kwargs)
    algo_instance.load(algorithm_path)
    
    # Set to eval mode
    for attr_name in dir(algo_instance):
        attr = getattr(algo_instance, attr_name)
        if isinstance(attr, nn.Module):
            attr.eval()
    
    # Ensure the algorithm instance has the environment set correctly
    algo_instance.env = env
    
    test_agent = create_agent_from_algorithm(algo_instance, agent_id=0, name="TrainedAgent", env=env)
    
    # Reset environment
    ep_seed = rng.randint(0, 1_000_000_000)
    observation, info = env.reset(seed=ep_seed)
    state = env.state
    
    # Create agents
    agents = {0: test_agent}
    for a_id in range(1, num_agents):
        agents[a_id] = opponent_bot_class(agent_id=a_id)
    
    # Get agent names
    agent_names = []
    for a_id in range(num_agents):
        if hasattr(agents[a_id], "name"):
            agent_names.append(agents[a_id].name)
        else:
            agent_names.append(f"Agent{a_id}")
    
    # Track states and actions
    states = [state]
    actions = []
    
    terminated = False
    while not terminated:
        # Compute actions - pass full observation dict to ensure action_mask is correct
        # Use a per-step seed derived from episode seed and step count for deterministic behavior
        step_seed = ep_seed + state.step_count if ep_seed is not None else None
        step_actions = []
        for a_id in range(num_agents):
            if state.alive[a_id]:
                # Update env state for action_mask if needed
                if hasattr(agents[a_id], 'env'):
                    agents[a_id].env = env
                # Pass the full observation dict, not just the grid
                # This ensures action_mask is properly set
                step_actions.append(agents[a_id](observation["grid"], seed=step_seed))
            else:
                # For dead agents, create a zero action grid of shape (H, W)
                H, W = grid_size
                step_actions.append(np.zeros((H, W), dtype=np.int32))
        
        action = np.stack(step_actions)
        actions.append(action)
        
        # Step environment
        observation, reward, terminated, truncated, info = env.step(action)
        state = env.state
        states.append(state)
        
        if truncated:
            terminated = True
    
    return states, actions, agent_names


def baseline_evaluation_with_stats_and_plots(
    algorithm_instance: Any,
    algorithm_config: Dict[str, Any],
    episodes: int = 100,
    seed: int = 0,
    grid_size: Tuple[int, int] = (30, 30),
    num_agents: int = 4,
    opponent_bot_classes: List[type] = None,
    verbose: bool = True,
    save_games: bool = True,
    games_output_dir: str = "evaluation_games"
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, List[Dict]], float, Dict[str, str]]:
    """
    Evaluates algorithm against different bot types and returns metrics and stats for plotting.
    Uses baseline_evaluation_with_stats from baseline_evaluation.py and extends it with stats collection and game saving.
    
    Returns:
        metrics_by_bot: Dictionary mapping bot names to metrics
        global_metrics: Overall metrics
        stats_list_by_bot: Dictionary mapping bot names to lists of stats dictionaries
        global_win_rate: Global win rate (float)
        saved_games: Dictionary mapping bot names to saved game file paths
    """
    from src.bots.rule_based import RuleBasedBot
    from src.bots.random_bot import RandomBot
    import tempfile
    import os
    from datetime import datetime
    
    if opponent_bot_classes is None:
        opponent_bot_classes = [RuleBasedBot, RandomBot]
    
    # Save algorithm to temporary file for parallel workers
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    temp_file.close()
    algorithm_path = temp_file.name
    
    try:
        algorithm_instance.save(algorithm_path)
    except Exception as e:
        os.unlink(algorithm_path)
        raise RuntimeError(f"Failed to save algorithm for evaluation: {e}")
    
    # Use baseline_evaluation_with_stats to get metrics
    metrics_by_bot, global_metrics = baseline_evaluation_with_stats(
        test_agent_cfg=None,
        episodes=episodes,
        seed=seed,
        grid_size=grid_size,
        num_agents=num_agents,
        opponent_bot_classes=opponent_bot_classes,
        algorithm_instance=algorithm_instance,
        algorithm_config=algorithm_config,
        verbose=verbose
    )
    
    # Extract global win rate
    global_win_rate = global_metrics.get('win_rate', {}).get('global', 0.0)
    
    # Collect stats_list_by_bot for plotting (using same seeds as baseline_evaluation_with_stats)
    stats_list_by_bot = {}
    saved_games = {}
    
    # Create output directory for games if saving
    if save_games:
        os.makedirs(games_output_dir, exist_ok=True)
    
    try:
        for bot_class in opponent_bot_classes:
            bot_name = bot_class.__name__
            
            # Use same seeds as baseline_evaluation_with_stats
            rng = np.random.RandomState(seed)
            seeds = [rng.randint(0, 1_000_000_000) for _ in range(episodes)]
            
            # Run in parallel to collect stats
            results = Parallel(n_jobs=-1)(
                delayed(_run_single_episode_with_stats)(
                    None, s, grid_size, num_agents, bot_class, 
                    None, algorithm_path, algorithm_config
                )
                for s in seeds
            )
            
            # Extract stats
            _, _, stats_list, _, _ = zip(*results)
            stats_list_by_bot[bot_name] = list(stats_list)
            
            # Save one game replay for this bot type
            if save_games:
                try:
                    game_seed = seeds[0]  # Use first seed for consistency
                    states, actions, agent_names = run_single_episode_for_replay(
                        algorithm_path=algorithm_path,
                        algorithm_config=algorithm_config,
                        seed=game_seed,
                        grid_size=grid_size,
                        num_agents=num_agents,
                        opponent_bot_class=bot_class
                    )
                    
                    # Create environment to use its save method
                    from src.env.env import Halite
                    env = Halite(grid_size=grid_size, num_agents=num_agents)
                    
                    # Generate filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    game_filename = f"{games_output_dir}/game_vs_{bot_name}_{timestamp}.hlt"
                    
                    # Save game
                    env.save(states, actions, player_names=agent_names, filepath=game_filename)
                    saved_games[bot_name] = game_filename
                    
                    if verbose:
                        print(f"  Saved game replay: {game_filename}")
                except Exception as e:
                    print(f"  Warning: Failed to save game replay for {bot_name}: {e}")
                    saved_games[bot_name] = None
        
    finally:
        # Clean up temporary file
        if os.path.exists(algorithm_path):
            try:
                os.unlink(algorithm_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary algorithm file {algorithm_path}: {e}")
    
    return metrics_by_bot, global_metrics, stats_list_by_bot, global_win_rate, saved_games


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a model and plot the results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model-folder',
        type=str,
        required=True,
        help='Path to model folder in models/ directory (e.g., models/mappo_B67C89)'
    )
    parser.add_argument(
        '--checkpoint-name',
        type=str,
        default=None,
        help='Name of checkpoint file in model folder (default: model.pt, or first .pt file found)'
    )
    parser.add_argument(
        '--eval-config',
        type=str,
        default=None,
        help='Path to evaluation config file (default: configs/evaluation/baseline_evaluation.yaml)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/plots/baseline_evaluation',
        help='Directory to save plots (default: outputs/plots/baseline_evaluation)'
    )
    parser.add_argument(
        '--agent-name',
        type=str,
        default=None,
        help='Name for the agent in plots (default: derived from checkpoint path)'
    )
    parser.add_argument(
        '--save-games',
        action='store_true',
        default=True,
        help='Save one game replay per opponent bot type (default: True)'
    )
    parser.add_argument(
        '--no-save-games',
        dest='save_games',
        action='store_false',
        help='Do not save game replays'
    )
    parser.add_argument(
        '--games-dir',
        type=str,
        default='evaluation_games',
        help='Directory to save game replays (default: evaluation_games)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load checkpoint and config from model folder
        algo, cfg, device = load_checkpoint_and_config(args.model_folder, args.checkpoint_name)
        
        # Load evaluation config
        # First check if main config has evaluation section
        if hasattr(cfg, 'evaluation') and cfg.evaluation is not None:
            print("Using evaluation config from main config file")
            eval_cfg = cfg.evaluation
        elif args.eval_config is not None:
            eval_config_path = Path(args.eval_config)
            if eval_config_path.exists():
                eval_cfg = OmegaConf.load(eval_config_path)
            else:
                print(f"Warning: Evaluation config not found at {eval_config_path}, using defaults")
                eval_cfg = OmegaConf.create({
                    'episodes': 10,
                    'opponent_bots': [
                        {'_target_': 'src.bots.rule_based.RuleBasedBot'},
                        {'_target_': 'src.bots.random_bot.RandomBot'}
                    ]
                })
        else:
            eval_config_path = Path('configs/evaluation/baseline_evaluation.yaml')
            if eval_config_path.exists():
                eval_cfg = OmegaConf.load(eval_config_path)
            else:
                print(f"Warning: Evaluation config not found at {eval_config_path}, using defaults")
                eval_cfg = OmegaConf.create({
                    'episodes': 10,
                    'opponent_bots': [
                        {'_target_': 'src.bots.rule_based.RuleBasedBot'},
                        {'_target_': 'src.bots.random_bot.RandomBot'}
                    ]
                })
        
        # Get evaluation parameters
        episodes = eval_cfg.get('episodes', 10)
        # Use seed from eval config, or derive from experiment seed, or default to 42
        experiment_seed = cfg.experiment.get('seed', cfg.get('seed', 42)) if hasattr(cfg, 'experiment') else cfg.get('seed', 42)
        seed = eval_cfg.get('seed', experiment_seed + 1000)  # Add 1000 to ensure different from training
        
        # Get grid_size and num_agents from training config or eval config
        grid_size = tuple(eval_cfg.get('grid_size', cfg.env.get('grid_size', [30, 30])))
        num_agents = eval_cfg.get('num_agents', cfg.env.get('num_agents', 4))
        
        # Get opponent bot classes
        opponent_bots = eval_cfg.get('opponent_bots', None)
        if opponent_bots is not None:
            opponent_bot_classes = [get_class(bot_cfg._target_) for bot_cfg in opponent_bots]
        else:
            from src.bots.rule_based import RuleBasedBot
            from src.bots.random_bot import RandomBot
            opponent_bot_classes = [RuleBasedBot, RandomBot]
        
        # Get algorithm config for reconstruction in workers
        algo_config = dict(OmegaConf.to_container(cfg.algo, resolve=True))
        algo_config['_target_'] = cfg.algo._target_
        algo_config['device'] = device
        
        print("\n" + "="*50)
        print("Evaluation Configuration:")
        print(f"  Episodes per bot type: {episodes}")
        print(f"  Grid size: {grid_size}")
        print(f"  Number of agents: {num_agents}")
        print(f"  Seed: {seed}")
        print(f"  Opponent bots: {[cls.__name__ for cls in opponent_bot_classes]}")
        print("="*50 + "\n")
        
        # Run evaluation
        metrics_by_bot, global_metrics, stats_list_by_bot, global_win_rate, saved_games = \
            baseline_evaluation_with_stats_and_plots(
                algorithm_instance=algo,
                algorithm_config=algo_config,
                episodes=episodes,
                seed=seed,
                grid_size=grid_size,
                num_agents=num_agents,
                opponent_bot_classes=opponent_bot_classes,
                verbose=True,
                save_games=args.save_games,
                games_output_dir=args.games_dir
            )
        
        # Extract win rates by bot
        win_rates_by_bot = {}
        for bot_name, bot_metrics in metrics_by_bot.items():
            win_rates_by_bot[bot_name] = bot_metrics.get('win_rate', {}).get('global', 0.0)
        
        # Print results
        print("\n" + "="*50)
        print("Evaluation Results:")
        print(f"Global Win Rate: {global_win_rate:.3f}")
        print("\nWin Rates by Bot Type:")
        for bot_name, win_rate in win_rates_by_bot.items():
            print(f"  {bot_name}: {win_rate:.3f}")
        print("="*50 + "\n")
        
        # Determine agent name (algorithm ID) from model folder
        if args.agent_name is None:
            model_folder = Path(args.model_folder)
            # Extract folder name (algorithm_ID)
            agent_name = model_folder.name
        else:
            agent_name = args.agent_name
        
        # Create plots
        print("Generating plots...")
        plot_file = save_halite_statistics_plot(
            win_rates_by_bot=win_rates_by_bot,
            stats_list_by_bot=stats_list_by_bot,
            global_win_rate=global_win_rate,
            player_idx=0,
            agent_name=agent_name,
            save_dir=args.output_dir
        )
        
        print(f"\n✓ Evaluation complete! Plot saved to: {plot_file}")
        
        # Print saved games
        if args.save_games and saved_games:
            print("\nSaved game replays:")
            for bot_name, game_path in saved_games.items():
                if game_path:
                    print(f"  vs {bot_name}: {game_path}")
                else:
                    print(f"  vs {bot_name}: (failed to save)")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

