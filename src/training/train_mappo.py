"""Training script for Multi-Agent PPO with Hydra configuration."""

import numpy as np
import torch
import hydra
import re
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from pathlib import Path
import secrets
import threading
from typing import Optional, Dict, List, Tuple

from src.env.env import Halite
from src.algo.mappo import MAPPO
from src.training.logger import Logger
from src.training.evaluation_utils import run_evaluation
from src.training.training_util import extract_local_observations, create_local_observation_dict

# Register custom resolver for random hex generation
# Check if already registered to avoid conflicts when importing from other scripts
try:
    OmegaConf.register_new_resolver("random_hex", lambda length: secrets.token_hex((int(length) + 1) // 2).upper()[:int(length)])
except ValueError:
    # Already registered, that's fine
    pass


@hydra.main(version_base=None, config_path="../../configs", config_name="train/train_mappo")
def train(cfg: DictConfig):
    """Train Multi-Agent PPO on Halite environment."""
    
    # Set random seeds for reproducibility
    # Get seed from experiment config (with fallback for backward compatibility)
    seed = cfg.experiment.get('seed', cfg.get('seed', 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Override device if specified in config
    device = cfg.get('device', 'cpu')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Experiment: {cfg.get('experiment_name', 'default')}")
    print(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create environment using Hydra instantiation
    env = instantiate(cfg.env)
    
    # Create algorithm - pass env and device directly, use config for other params
    algo_kwargs = OmegaConf.to_container(cfg.algo, resolve=True)
    # Remove use_local_observations if present (no longer used, always uses local observations)
    algo_kwargs.pop('use_local_observations', None)
    algo_kwargs['env'] = env
    algo_kwargs['device'] = device
    # Remove _target_ before instantiating
    target_class = algo_kwargs.pop('_target_')
    from hydra.utils import get_class
    algo_class = get_class(target_class)
    algo = algo_class(**algo_kwargs)
    
    # Get experiment config
    exp_cfg = cfg.experiment
    exp_name = cfg.get('experiment_name', 'default')
    
    # Create experiment directory structure
    experiment_dir = Path('outputs') / 'experiments' / exp_name
    checkpoint_dir = experiment_dir / 'checkpoints'
    sample_games_dir = experiment_dir / 'sample_games'
    
    # Create all directories
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_games_dir.mkdir(parents=True, exist_ok=True)
    
    # Create final model directory with experiment name
    final_model_dir = Path('models') / exp_name
    final_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = Logger(cfg, env, 'MAPPO')
    
    # Setup evaluation if configured
    eval_config = cfg.get('evaluation', None)
    eval_thread: Optional[threading.Thread] = None
    eval_results = {}
    
    # Track if config has been saved
    config_saved = False
    
    # Resume from checkpoint if specified
    start_episode = 0
    if exp_cfg.get('resume_from_checkpoint', False):
        resume_experiment_name = exp_cfg.get('resume_experiment_name')
        resume_checkpoint_name = exp_cfg.get('resume_checkpoint_name')
        
        if resume_experiment_name is None or resume_checkpoint_name is None:
            raise ValueError("resume_from_checkpoint is True but resume_experiment_name or resume_checkpoint_name is not specified")
        
        # Construct checkpoint path
        resume_experiment_dir = Path('outputs') / 'experiments' / resume_experiment_name
        resume_checkpoint_path = resume_experiment_dir / 'checkpoints' / resume_checkpoint_name
        
        if not resume_checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_checkpoint_path}")
        
        # Extract episode number from checkpoint filename (e.g., "ep5000.pt" -> 5000)
        match = re.search(r'ep(\d+)\.pt', resume_checkpoint_name)
        if match:
            start_episode = int(match.group(1)) + 1
        else:
            raise ValueError(f"Could not extract episode number from checkpoint filename: {resume_checkpoint_name}")
        
        # Load checkpoint
        print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        print(f"Starting from episode: {start_episode}")
        algo.load(str(resume_checkpoint_path))
        
        # Update curriculum reward to the correct episode if applicable
        if hasattr(env._reward_fn, 'update_episode'):
            env._reward_fn.update_episode(start_episode)
        
        # Mark config as saved since we're resuming
        config_saved = True
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(start_episode, exp_cfg.num_episodes):
        # Update curriculum reward if applicable
        if hasattr(env._reward_fn, 'update_episode'):
            env._reward_fn.update_episode(episode)
        
        # Use episode-specific seed derived from experiment seed for reproducibility
        episode_seed = seed + episode
        obs, _ = env.reset(seed=episode_seed)
        episode_reward = np.zeros(env.num_agents)
        episode_length = 0
        done = False
        update_stats = None
        
        # Track states and actions for saving replays
        episode_states = [env.state]  # Save initial state
        episode_actions = []
        
        while not done:
            # Create local observations for each agent's units
            # Each unit gets a 7x7 patch with 6 channels:
            # [is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask]
            local_obs_dict = create_local_observation_dict(obs, env.num_agents)
            
            # Select action (always uses local observations)
            action, log_prob, value = algo.select_action(obs, local_obs_dict)
            
            # Save action before stepping
            episode_actions.append(action.copy())
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Save state after step
            episode_states.append(env.state)
            
            # Train step (collects data in episode buffer, updates when done)
            # Uses local observations for actor, global observations for critic
            update_stats = algo.train_step(obs, action, reward, next_obs, done, log_prob, value)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log metrics to wandb
        logger.log_episode_metrics(episode + 1, episode_reward, episode_length, env.state, update_stats)
        
        # Print progress
        if (episode + 1) % exp_cfg.log_frequency == 0:
            avg_reward = np.mean([np.sum(r) for r in episode_rewards[-exp_cfg.log_frequency:]])
            avg_length = np.mean(episode_lengths[-exp_cfg.log_frequency:])
            print(f"Episode {episode + 1}/{exp_cfg.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
            if update_stats:
                print(f"  Actor Loss: {update_stats.get('actor_loss', 0):.4f}")
                print(f"  Critic Loss: {update_stats.get('critic_loss', 0):.4f}")
                print(f"  Entropy: {update_stats.get('entropy', 0):.4f}")
        
        # Save checkpoint
        if (episode + 1) % exp_cfg.save_frequency == 0:
            # Save config YAML on first checkpoint
            if not config_saved:
                config_path = experiment_dir / 'config.yaml'
                with open(config_path, 'w') as f:
                    f.write(OmegaConf.to_yaml(cfg))
                print(f"Config saved to {config_path}")
                config_saved = True
            
            checkpoint_path = checkpoint_dir / f"ep{episode+1}.pt"
            algo.save(str(checkpoint_path))
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Save sample game replay with same naming as checkpoint
            try:
                player_names = [f"MAPPO_Agent_{i}" for i in range(env.num_agents)]
                game_path = sample_games_dir / f"ep{episode+1}.hlt"
                env.save(episode_states, episode_actions, player_names=player_names, filepath=str(game_path))
                print(f"[Training] Saved game replay for episode {episode + 1}")
            except Exception as e:
                print(f"[Training] Warning: Failed to save game replay: {e}")
        
        # Run evaluation asynchronously
        if eval_config is not None and (episode + 1) % exp_cfg.eval_frequency == 0:
            # Wait for previous evaluation to finish if still running
            if eval_thread is not None and eval_thread.is_alive():
                print(f"[Evaluation] Waiting for previous evaluation to finish...")
                eval_thread.join()
            
            # Start new evaluation in background
            eval_thread = threading.Thread(
                target=run_evaluation,
                args=(episode + 1, algo, eval_config, env, cfg, device, logger, eval_results),
                daemon=True
            )
            eval_thread.start()
    
    # Wait for any running evaluation to finish
    if eval_thread is not None and eval_thread.is_alive():
        print("\n[Evaluation] Waiting for final evaluation to finish...")
        eval_thread.join()
    
    # Run final evaluation only if it doesn't coincide with the last evaluation
    if eval_config is not None:
        # Check if an evaluation was already run at the last episode
        # Evaluations are triggered when (episode + 1) % eval_frequency == 0
        # So if num_episodes % eval_frequency == 0, an evaluation was already run
        if exp_cfg.num_episodes % exp_cfg.eval_frequency != 0:
            print("\n[Evaluation] Running final evaluation...")
            run_evaluation(exp_cfg.num_episodes, algo, eval_config, env, cfg, device, logger, eval_results)
        else:
            print(f"\n[Evaluation] Final evaluation skipped (already evaluated at episode {exp_cfg.num_episodes})")
    
    # Save final model and config
    # Ensure directory exists (in case it was deleted or path changed)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model_path = final_model_dir / 'model.pt'
    algo.save(str(model_path))
    print(f"\nFinal model saved to {model_path}")
    
    # Save config to final model directory
    final_config_path = final_model_dir / 'config.yaml'
    with open(final_config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    print(f"Final config saved to {final_config_path}")
    
    # Finish logging
    logger.finish()
    
    return algo, episode_rewards, episode_lengths


if __name__ == '__main__':
    train()
