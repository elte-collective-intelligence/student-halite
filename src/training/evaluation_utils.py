"""Shared evaluation utilities for training scripts."""

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, List, Type

from src.eval.baseline_evaluation import baseline_evaluation_with_stats
from src.bots.rule_based import RuleBasedBot
from src.bots.random_bot import RandomBot
from src.bots.rule_based_v1 import RuleBasedV1
from src.bots.rule_based_v2 import RuleBasedV2


def set_eval_mode(algo_instance, mode: bool = True):
    """Set all networks in algorithm to eval/train mode.
    
    Args:
        algo_instance: The algorithm instance containing neural networks
        mode: If True, set to eval mode; if False, set to train mode
    """
    for attr_name in dir(algo_instance):
        attr = getattr(algo_instance, attr_name)
        if isinstance(attr, nn.Module):
            if mode:
                attr.eval()
            else:
                attr.train()


def run_evaluation(
    episode_num: int,
    algo_instance: Any,
    eval_cfg: DictConfig,
    env: Any,
    cfg: DictConfig,
    device: str,
    logger: Any,
    eval_results: Dict[int, Dict[str, Any]]
):
    """Run evaluation in background thread.
    
    Args:
        episode_num: Current episode number
        algo_instance: The algorithm instance to evaluate
        eval_cfg: Evaluation configuration
        env: Training environment instance
        cfg: Full training configuration
        device: Device to use ('cpu' or 'cuda')
        logger: Logger instance for logging metrics
        eval_results: Dictionary to store evaluation results
    """
    try:
        print(f"\n[Evaluation] Starting evaluation at episode {episode_num}...")
        
        # Get evaluation parameters
        episodes = eval_cfg.get('episodes', 20)
        # Use grid_size and num_agents from training environment, not evaluation config
        grid_size = env.grid_size
        num_agents = env.num_agents
        # Get seed: prefer eval_cfg seed, otherwise derive from experiment seed for reproducibility
        # Using experiment_seed + 1000 ensures evaluation is reproducible but separate from training
        experiment_seed = cfg.experiment.get('seed', cfg.get('seed', 42))
        seed = eval_cfg.get('seed', experiment_seed + 1000)
        
        # Get opponent bots
        opponent_bots = eval_cfg.get('opponent_bots', None)
        opponent_bot_classes: List[Type] = [RandomBot, RuleBasedBot, RuleBasedV1, RuleBasedV2]  # Default
        if opponent_bots is not None and len(opponent_bots) > 0:
            from hydra.utils import get_class
            print(f"[Evaluation] Loading {len(opponent_bots)} opponent bot(s) from config")
            try:
                opponent_bot_classes = []
                for bot_cfg in opponent_bots:
                    try:
                        bot_class = get_class(bot_cfg._target_)
                        opponent_bot_classes.append(bot_class)
                        print(f"[Evaluation]   Loaded: {bot_cfg._target_} -> {bot_class.__name__}")
                    except Exception as e:
                        print(f"[Evaluation]   Warning: Failed to load bot class {bot_cfg._target_}: {e}")
                
                # Fall back to default if list comprehension resulted in empty list
                if len(opponent_bot_classes) == 0:
                    print(f"[Evaluation] Warning: No valid bot classes loaded, using default [RuleBasedBot, RandomBot, RuleBasedV1, RuleBasedV2]")
                    opponent_bot_classes = [RandomBot, RuleBasedBot, RuleBasedV1, RuleBasedV2]
                else:
                    print(f"[Evaluation] Using {len(opponent_bot_classes)} opponent bot class(es): {[cls.__name__ for cls in opponent_bot_classes]}")
            except Exception as e:
                print(f"[Evaluation] Error loading opponent bots: {e}, using default [RuleBasedBot, RandomBot, RuleBasedV1, RuleBasedV2]")
                opponent_bot_classes = [RandomBot, RuleBasedBot, RuleBasedV1, RuleBasedV2]
        else:
            print(f"[Evaluation] No opponent_bots in config (got {opponent_bots}), using default [RuleBasedBot, RandomBot, RuleBasedV1, RuleBasedV2]")
        
        # Set algorithm to eval mode
        set_eval_mode(algo_instance, mode=True)
        
        # Get algorithm config for reconstruction in workers
        # Need to preserve _target_ from original config
        algo_config = dict(OmegaConf.to_container(cfg.algo, resolve=True))
        algo_config['_target_'] = cfg.algo._target_  # Always include _target_
        algo_config['device'] = device
        
        # Run evaluation (verbose=False to reduce output during training)
        metrics_by_bot, global_metrics = baseline_evaluation_with_stats(
            episodes=episodes,
            seed=seed,
            grid_size=grid_size,
            num_agents=num_agents,
            opponent_bot_classes=opponent_bot_classes,
            algorithm_instance=algo_instance,
            algorithm_config=algo_config,
            verbose=False
        )
        
        # Set back to train mode
        set_eval_mode(algo_instance, mode=False)
        
        # Log metrics at the episode when evaluation started
        # This ensures evaluation data is associated with the episode it was triggered at
        logger.log_evaluation_metrics(episode_num, global_metrics)
        
        # Also log per-bot metrics with bot name as prefix
        for bot_name, bot_metrics in metrics_by_bot.items():
            logger.log_evaluation_metrics(episode_num, bot_metrics, prefix=bot_name)
        
        eval_results[episode_num] = {
            'by_bot': metrics_by_bot,
            'global': global_metrics
        }
        
        print(f"[Evaluation] Completed evaluation (started and logged at episode {episode_num})")
        print(f"  Global win rate: {global_metrics.get('win_rate', {}).get('global', 0):.3f}")
        print(f"  Per-bot win rates:")
        for bot_name, bot_metrics in metrics_by_bot.items():
            bot_win_rate = bot_metrics.get('win_rate', {}).get('global', 0.0)
            print(f"    {bot_name}: {bot_win_rate:.3f}")
        
    except Exception as e:
        print(f"[Evaluation] Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

