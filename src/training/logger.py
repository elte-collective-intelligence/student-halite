"""Logger class for experiment monitoring with Weights & Biases and local storage."""

import os
import wandb
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import DictConfig, OmegaConf
import numpy as np


class Logger:
    """Logger for tracking experiments with wandb and local storage."""
    
    def __init__(
        self,
        cfg: DictConfig,
        env,
        algo_name: str,
        wb_config: Optional[DictConfig] = None
    ):
        """Initialize the logger.
        
        Args:
            cfg: Full Hydra configuration
            env: Environment instance
            algo_name: Name of the algorithm (e.g., 'IQL', 'IPPO', 'MAPPO', 'CQL')
            wb_config: Optional wandb configuration (if None, tries to get from cfg.experiment.wb)
        """
        self.cfg = cfg
        self.env = env
        self.algo_name = algo_name
        
        # Try to get wb_config from cfg if not provided
        if wb_config is None:
            try:
                wb_config = OmegaConf.select(cfg, 'experiment.wb', default=None)
            except (AttributeError, KeyError):
                wb_config = None
        
        self.wb_config = wb_config
        self.wandb_run = None
        self.current_episode = 0  # Track current episode for async evaluation logging
        self.use_console_logging = False  # Flag to enable console logging when wandb is disabled
        
        # Initialize wandb if config is provided
        had_wb_config = wb_config is not None
        if wb_config is not None:
            self._init_wandb()
        
        # If wandb was not initialized (but we had a config), enable console logging
        if self.wandb_run is None and had_wb_config:
            self.use_console_logging = True
            print("Logging metrics to console (wandb API key not available).")
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        # Get API key from config or environment variable
        wb_dict = OmegaConf.to_container(self.wb_config, resolve=True) if self.wb_config else {}
        api_key = wb_dict.get('api_key') or os.getenv('WANDB_API_KEY')
        
        # Handle None/null values
        if api_key is None or api_key == 'null' or api_key == '':
            api_key = None
        
        if api_key:
            os.environ['WANDB_API_KEY'] = api_key
        elif not os.getenv('WANDB_API_KEY'):
            print("Warning: No wandb API key found. Wandb logging will be disabled.")
            self.wb_config = None
            return
        
        # Get project and entity
        wb_dict = OmegaConf.to_container(self.wb_config, resolve=True) if self.wb_config else {}
        project = wb_dict.get('project_name', 'student-halite')
        entity = wb_dict.get('entity') or os.getenv('WANDB_ENTITY')
        
        # Initialize wandb run
        # If resuming from checkpoint, use the resume experiment name to continue the same wandb run
        exp_cfg = self.cfg.get('experiment', {})
        if exp_cfg.get('resume_from_checkpoint', False):
            resume_experiment_name = exp_cfg.get('resume_experiment_name')
            if resume_experiment_name:
                # Use the resume experiment name to continue the original wandb run
                run_name = resume_experiment_name
                print(f"Resuming wandb run for experiment: {resume_experiment_name}")
            else:
                run_name = self.cfg.get('experiment_name', f'{self.algo_name}_experiment')
        else:
            run_name = self.cfg.get('experiment_name', f'{self.algo_name}_experiment')
        
        # Extract ID from experiment name to align with directory structure
        # Format is typically: {prefix}_{hex_id} (e.g., mappo_B67C89)
        # Extract the hex_id part and use it as wandb run ID
        run_id = None
        if '_' in run_name:
            # Split on last underscore to get the ID part
            parts = run_name.rsplit('_', 1)
            if len(parts) == 2:
                potential_id = parts[1]
                # Check if it looks like a hex ID (alphanumeric, typically 6 chars)
                if potential_id.isalnum() and len(potential_id) >= 4:
                    # Convert to lowercase as wandb IDs are typically lowercase
                    run_id = potential_id.lower()
        
        # When resuming, use resume=True to continue the existing run instead of creating a new one
        resume_wandb = exp_cfg.get('resume_from_checkpoint', False) and run_id is not None
        
        try:
            self.wandb_run = wandb.init(
                project=project,
                entity=entity if entity else None,
                name=run_name,
                id=run_id,  # Set explicit ID to match directory structure
                resume="allow" if resume_wandb else None,  # Allow resuming existing run if resuming from checkpoint
                config=self._get_wandb_config(),
                reinit=True
            )
            
            # Define evaluation metrics to allow out-of-order logging
            # This prevents warnings when evaluation runs asynchronously
            # All eval metrics are independent of the step counter
            eval_metric_prefixes = [
                'eval/win_rate', 'eval/avg_position', 'eval/time_lasted',
                'eval/max_territory', 'eval/max_strength', 'eval/max_production',
                'eval/cap_losses', 'eval/successful_captures', 'eval/engagement_efficiency',
                'eval_triggered_at', 'evaluation_episode'
            ]
            # Define each metric prefix to not use step tracking
            # Note: We don't define 'episode' here as it's also used in training metrics
            for prefix in eval_metric_prefixes:
                try:
                    wandb.define_metric(prefix, step_metric=None)
                except Exception:
                    # If pattern-based definition fails, we'll define metrics individually when logging
                    pass
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}. Wandb logging will be disabled.")
            self.wandb_run = None
            self.wb_config = None
    
    def _sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively remove sensitive information (API keys) from config."""
        if not isinstance(config, dict):
            return config
        
        sanitized = {}
        for key, value in config.items():
            # Skip API key at any level
            if key == 'api_key':
                continue
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self._sanitize_config(value)
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples that might contain dicts
                sanitized[key] = [
                    self._sanitize_config(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _get_wandb_config(self) -> Dict[str, Any]:
        """Convert Hydra config to wandb-compatible format."""
        config_dict = {}
        
        # Environment details
        env_config = {
            'grid_size': self.cfg.env.get('grid_size', [25, 25]),
            'num_agents': self.cfg.env.get('num_agents', 4),
        }
        
        # Extract reward function details
        if hasattr(self.cfg.env, 'reward_fn'):
            reward_fn_cfg = self.cfg.env.reward_fn
            reward_fn_target = reward_fn_cfg.get('_target_', 'unknown')
            env_config['reward_fn'] = reward_fn_target
            
            # If using curriculum-shaped reward, extract all its parameters
            if 'curriculum_shaped' in reward_fn_target.lower():
                curriculum_params = {}
                
                # Extract main curriculum parameters
                if 'schedule_type' in reward_fn_cfg:
                    curriculum_params['schedule_type'] = reward_fn_cfg['schedule_type']
                if 'start_episode' in reward_fn_cfg:
                    curriculum_params['start_episode'] = reward_fn_cfg['start_episode']
                if 'end_episode' in reward_fn_cfg:
                    curriculum_params['end_episode'] = reward_fn_cfg['end_episode']
                if 'start_weight' in reward_fn_cfg:
                    curriculum_params['start_weight'] = reward_fn_cfg['start_weight']
                if 'end_weight' in reward_fn_cfg:
                    curriculum_params['end_weight'] = reward_fn_cfg['end_weight']
                
                # Extract schedule-specific parameters
                if 'schedule_kwargs' in reward_fn_cfg:
                    schedule_kwargs = OmegaConf.to_container(reward_fn_cfg['schedule_kwargs'], resolve=True)
                    if schedule_kwargs:
                        curriculum_params['schedule_kwargs'] = schedule_kwargs
                
                # Extract shaped reward function parameters
                if 'shaped_reward_fn' in reward_fn_cfg:
                    shaped_cfg = reward_fn_cfg['shaped_reward_fn']
                    shaped_params = {
                        '_target_': shaped_cfg.get('_target_', 'unknown')
                    }
                    # Extract all shaped reward parameters
                    for key in ['territory_weight', 'strength_weight', 'production_weight', 
                               'expansion_bonus', 'loss_penalty_multiplier', 
                               'zero_strength_move_penalty', 'normalize']:
                        if key in shaped_cfg:
                            shaped_params[key] = shaped_cfg[key]
                    curriculum_params['shaped_reward_fn'] = shaped_params
                
                # Extract minimal reward function parameters
                if 'minimal_reward_fn' in reward_fn_cfg:
                    minimal_cfg = reward_fn_cfg['minimal_reward_fn']
                    curriculum_params['minimal_reward_fn'] = {
                        '_target_': minimal_cfg.get('_target_', 'unknown')
                    }
                
                # Add curriculum parameters to env config
                env_config['curriculum_reward'] = curriculum_params
            else:
                # For non-curriculum rewards, extract basic parameters if available
                reward_params = {}
                for key in ['territory_weight', 'strength_weight', 'production_weight',
                           'expansion_bonus', 'loss_penalty_multiplier',
                           'zero_strength_move_penalty', 'normalize', 'reward_scale']:
                    if key in reward_fn_cfg:
                        reward_params[key] = reward_fn_cfg[key]
                if reward_params:
                    env_config['reward_params'] = reward_params
        else:
            env_config['reward_fn'] = 'unknown'
        
        config_dict['env'] = env_config
        
        # Algorithm details
        algo_cfg = OmegaConf.to_container(self.cfg.algo, resolve=True)
        if '_target_' in algo_cfg:
            algo_cfg['name'] = algo_cfg.pop('_target_')
        config_dict['algo'] = algo_cfg
        
        # Experiment details - sanitize to remove API key (especially experiment.wb.api_key)
        exp_cfg = OmegaConf.to_container(self.cfg.experiment, resolve=True)
        config_dict['experiment'] = self._sanitize_config(exp_cfg)
        
        # Additional safety: explicitly remove api_key from wb if present
        if 'experiment' in config_dict and 'wb' in config_dict['experiment']:
            if 'api_key' in config_dict['experiment']['wb']:
                del config_dict['experiment']['wb']['api_key']
        # Get seed from experiment config (with fallback for backward compatibility)
        config_dict['seed'] = self.cfg.experiment.get('seed', self.cfg.get('seed', 42))
        config_dict['device'] = self.cfg.get('device', 'auto')
        
        return config_dict
    
    def log_metrics(self, episode: int, metrics: Dict[str, Any]):
        """Log metrics to wandb or console.
        
        Args:
            episode: Current episode number
            metrics: Dictionary of metrics to log (e.g., {'territory': [0.25, 0.25, 0.25, 0.25]})
        """
        log_dict = {'episode': episode}
        
        # Log territory metrics
        if 'territory' in metrics:
            territory = metrics['territory']
            if isinstance(territory, (list, np.ndarray)):
                # Log per-agent territory
                for i, terr in enumerate(territory):
                    log_dict[f'territory/agent_{i}'] = float(terr)
                # Log average territory
                log_dict['territory/mean'] = float(np.mean(territory))
                # Log max territory
                log_dict['territory/max'] = float(np.max(territory))
            else:
                log_dict['territory'] = float(territory)
        
        # Log any other metrics
        for key, value in metrics.items():
            if key != 'territory':
                if isinstance(value, (list, np.ndarray)):
                    for i, v in enumerate(value):
                        log_dict[f'{key}/agent_{i}'] = float(v)
                    log_dict[f'{key}/mean'] = float(np.mean(value))
                else:
                    log_dict[key] = float(value)
        
        # Log to wandb if available
        if self.wandb_run:
            wandb.log(log_dict, step=episode)
        # Log to console if wandb is not available
        elif self.use_console_logging:
            self._log_to_console(episode, log_dict)
        
        # Update current episode to ensure evaluation metrics use correct step
        self.current_episode = episode
    
    def _log_to_console(self, episode: int, log_dict: Dict[str, Any]):
        """Log metrics to console in a readable format.
        
        Args:
            episode: Current episode number
            log_dict: Dictionary of metrics to log
        """
        # Format metrics for console output
        lines = []
        for key, value in sorted(log_dict.items()):
            if key != 'episode':
                if isinstance(value, (int, float)):
                    lines.append(f"    {key}: {value:.4f}")
                else:
                    lines.append(f"    {key}: {value}")
        if lines:
            print(f"[Episode {episode}] Metrics:")
            print('\n'.join(lines))
    
    def log_episode_metrics(
        self,
        episode: int,
        episode_reward: np.ndarray,
        episode_length: int,
        final_state=None,
        update_stats=None
    ):
        """Log episode metrics including territory and loss.
        
        Args:
            episode: Current episode number
            episode_reward: Array of rewards for each agent
            episode_length: Length of the episode
            final_state: Final state of the episode (optional, for territory calculation)
            update_stats: Training statistics from algorithm (dict or list of dicts)
        """
        # Continue logging even if wandb is not available (will log to console)
        
        metrics = {}
        
        # Calculate territory from final state if available
        if final_state is not None:
            territory = self._calculate_territory(final_state)
            metrics['territory'] = territory
        
        # Log curriculum reward weight if using curriculum-shaped reward
        if hasattr(self.env, '_reward_fn') and hasattr(self.env._reward_fn, 'get_current_weight'):
            try:
                current_weight = self.env._reward_fn.get_current_weight()
                metrics['curriculum/weight'] = float(current_weight)
            except Exception:
                # If weight calculation fails, skip it
                pass
        
        # Log rewards
        if isinstance(episode_reward, np.ndarray):
            for i, reward in enumerate(episode_reward):
                metrics[f'reward/agent_{i}'] = float(reward)
            metrics['reward/mean'] = float(np.mean(episode_reward))
            metrics['reward/sum'] = float(np.sum(episode_reward))
        
        # Log episode length
        metrics['episode_length'] = float(episode_length)
        
        # Log loss information if available
        if update_stats is not None:
            if isinstance(update_stats, list):
                # Handle list of stats (e.g., IQL, IPPO with per-agent stats)
                valid_stats = [s for s in update_stats if s is not None and isinstance(s, dict)]
                if valid_stats:
                    # Aggregate losses across agents
                    loss_keys = set()
                    for stat in valid_stats:
                        loss_keys.update(stat.keys())
                    
                    for key in loss_keys:
                        values = [stat.get(key, 0) for stat in valid_stats]
                        if values:
                            metrics[f'loss/{key}/mean'] = float(np.mean(values))
                            metrics[f'loss/{key}/sum'] = float(np.sum(values))
                            # Log per-agent if multiple agents
                            if len(valid_stats) > 1:
                                for i, stat in enumerate(valid_stats):
                                    if key in stat:
                                        metrics[f'loss/{key}/agent_{i}'] = float(stat[key])
            elif isinstance(update_stats, dict):
                # Handle dict of stats (e.g., MAPPO, CQL)
                for key, value in update_stats.items():
                    metrics[f'loss/{key}'] = float(value)
        
        self.log_metrics(episode, metrics)
    
    def _calculate_territory(self, state) -> np.ndarray:
        """Calculate territory share for each agent from state.
        
        Args:
            state: State object with grid attribute
            
        Returns:
            Array of territory shares (normalized 0-1) for each agent
        """
        if not hasattr(state, 'grid'):
            return np.zeros(self.env.num_agents)
        
        owner_map = state.grid[0]  # Ownership layer
        H, W = owner_map.shape
        total_cells = H * W
        
        territory = np.zeros(self.env.num_agents, dtype=np.float32)
        for agent_id in range(self.env.num_agents):
            player_id = agent_id + 1
            territory[agent_id] = np.sum(owner_map == player_id) / total_cells
        
        return territory
    
    def log_evaluation_metrics(self, episode: int, metrics: Dict[str, Any], prefix: str = ''):
        """Log evaluation metrics from baseline evaluation.
        
        Args:
            episode: Episode number when evaluation was triggered
            metrics: Dictionary of evaluation metrics from baseline_evaluation_with_stats
            prefix: Optional prefix for metric keys (e.g., bot name for per-bot metrics)
        """
        # Continue logging even if wandb is not available (will log to console)
        
        # Store the episode when evaluation was triggered - this is critical for async evaluation
        # where training may have progressed beyond this episode by the time evaluation completes
        eval_triggered_episode = episode
        
        # Include episode when evaluation was triggered (for reference/filtering)
        # Register evaluation episode as an additional metric to ensure it's properly tracked
        # The step will use episode to store evaluation data on the episode it starts
        log_dict = {
            'episode': eval_triggered_episode,  # Episode when evaluation was triggered
            'eval_triggered_at': eval_triggered_episode,  # Explicit field for when eval started (preserved)
            'evaluation_episode': eval_triggered_episode  # Additional metric to register evaluation episode
        }
        
        # For per-bot metrics (with prefix), only log win rates
        # For global metrics (no prefix), log all metrics
        if prefix:
            # Per-bot metrics: only log win rate
            if 'win_rate' in metrics:
                wr = metrics['win_rate']
                if 'global' in wr:
                    log_dict[f'eval/{prefix}/win_rate'] = float(wr['global'])
        else:
            # Global metrics: log all metrics
            # Win rates
            if 'win_rate' in metrics:
                wr = metrics['win_rate']
                if 'global' in wr:
                    log_dict['eval/win_rate/global'] = float(wr['global'])
                if 'by_opponent' in wr:
                    for opp, rate in wr['by_opponent'].items():
                        # Skip 'global' key in by_opponent as it's redundant
                        if opp != 'global':
                            log_dict[f'eval/win_rate/{opp}'] = float(rate)
            
            # Average position
            if 'avg_position' in metrics:
                log_dict['eval/avg_position'] = float(metrics['avg_position'])
            
            # Time lasted
            if 'time_lasted' in metrics:
                tl = metrics['time_lasted']
                for key in ['avg', 'max', 'min', 'std']:
                    if key in tl:
                        log_dict[f'eval/time_lasted/{key}'] = float(tl[key])
            
            # Max territory
            if 'max_territory' in metrics:
                mt = metrics['max_territory']
                for key in ['avg', 'std']:
                    if key in mt:
                        log_dict[f'eval/max_territory/{key}'] = float(mt[key])
            
            # Max strength
            if 'max_strength' in metrics:
                ms = metrics['max_strength']
                for key in ['avg', 'std']:
                    if key in ms:
                        log_dict[f'eval/max_strength/{key}'] = float(ms[key])
            
            # Max production
            if 'max_production' in metrics:
                mp = metrics['max_production']
                for key in ['avg', 'std']:
                    if key in mp:
                        log_dict[f'eval/max_production/{key}'] = float(mp[key])
            
            # Cap losses
            if 'cap_losses' in metrics:
                cl = metrics['cap_losses']
                for key in ['avg', 'max', 'min', 'std']:
                    if key in cl:
                        log_dict[f'eval/cap_losses/{key}'] = float(cl[key])
            
            # Successful captures
            if 'successful_captures' in metrics:
                sc = metrics['successful_captures']
                for key in ['avg', 'max', 'min', 'std']:
                    if key in sc:
                        log_dict[f'eval/successful_captures/{key}'] = float(sc[key])
            
            # Engagement efficiency
            if 'engagement_efficiency' in metrics:
                log_dict['eval/engagement_efficiency'] = float(metrics['engagement_efficiency'])
        
        # Log evaluation metrics - episode is included as a metric in log_dict
        # Log to wandb if available
        if self.wandb_run:
            # Define all evaluation metrics to allow out-of-order logging (step_metric=None)
            # This prevents warnings when evaluation runs asynchronously
            # We define metrics here to ensure they're independent of the step counter
            for key in log_dict.keys():
                try:
                    # Define each metric to not use step tracking
                    # This allows evaluation metrics to be logged out of order
                    wandb.define_metric(key, step_metric=None)
                except Exception:
                    # Metric might already be defined, continue
                    pass
            
            # Log without step parameter - metrics are defined to be step-independent
            # This prevents wandb from trying to use step tracking for evaluation metrics
            wandb.log(log_dict)
        # Log to console if wandb is not available
        elif self.use_console_logging:
            lines = []
            for key, value in sorted(log_dict.items()):
                if key not in ['episode', 'eval_triggered_at', 'evaluation_episode']:
                    if isinstance(value, (int, float)):
                        lines.append(f"    {key}: {value:.4f}")
                    else:
                        lines.append(f"    {key}: {value}")
            if lines:
                print(f"[Evaluation at Episode {episode}] Metrics:")
                print('\n'.join(lines))
    
    def finish(self):
        """Finish logging and close wandb run."""
        if self.wandb_run:
            wandb.finish()
            self.wandb_run = None

