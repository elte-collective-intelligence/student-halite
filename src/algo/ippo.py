"""Independent Proximal Policy Optimization (IPPO) for multi-agent Halite."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict

from src.algo.networks import LocalActorNetwork, CriticNetwork
from src.algo.replay_buffer import EpisodeBuffer
from src.env.env import Halite
from src.training.training_util import extract_local_observations


class IPPO:
    """Independent PPO with per-agent actors and critics."""
    
    def __init__(
        self,
        env: Halite,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        shared_weights: bool = False,
        device: str = 'cpu'
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.shared_weights = shared_weights
        self.device = device
        
        grid_size = env.grid_size
        num_agents = env.num_agents
        
        # Initialize per-agent actors and critics
        # Actors use local 7x7 observations
        if shared_weights:
            # Shared network architecture (weights shared across agents)
            self.actor_networks = LocalActorNetwork(
                input_channels=6,  # 6 channels: is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask
                patch_size=7,
                num_actions=5
            ).to(device)
            self.critic_networks = CriticNetwork(
                input_channels=3,
                grid_size=grid_size,
                centralized=False,
                num_agents=1
            ).to(device)
        else:
            # Independent networks for each agent
            self.actor_networks = nn.ModuleList([
                LocalActorNetwork(
                    input_channels=6,
                    patch_size=7,
                    num_actions=5
                ).to(device) for _ in range(num_agents)
            ])
            self.critic_networks = nn.ModuleList([
                CriticNetwork(
                    input_channels=3,
                    grid_size=grid_size,
                    centralized=False,
                    num_agents=1
                ).to(device) for _ in range(num_agents)
            ])
        
        # Optimizers
        if shared_weights:
            self.actor_optimizers = optim.Adam(self.actor_networks.parameters(), lr=lr_actor)
            self.critic_optimizers = optim.Adam(self.critic_networks.parameters(), lr=lr_critic)
        else:
            self.actor_optimizers = [
                optim.Adam(net.parameters(), lr=lr_actor) for net in self.actor_networks
            ]
            self.critic_optimizers = [
                optim.Adam(net.parameters(), lr=lr_critic) for net in self.critic_networks
            ]
        
        # Episode buffers (one per agent)
        self.episode_buffers = [EpisodeBuffer() for _ in range(num_agents)]
        
        self.training_stats = defaultdict(list)
    
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions using actor networks with local observations.
        
        Args:
            obs: Observation dictionary
            seed: Optional random seed for deterministic action sampling
            
        Returns:
            actions: (num_agents, height, width)
            log_probs: (num_agents, height, width) - log probabilities
            values: (num_agents,) - estimated values
        """
        grid = obs['grid']  # (3, H, W)
        action_mask = obs['action_mask']  # (num_agents, H, W, 5)
        
        # Get actual grid dimensions
        H, W = grid.shape[1], grid.shape[2]
        
        actions = np.zeros((self.env.num_agents, H, W), dtype=np.int32)
        log_probs = np.zeros((self.env.num_agents, H, W), dtype=np.float32)
        values = np.zeros(self.env.num_agents, dtype=np.float32)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
            
            for agent_id in range(self.env.num_agents):
                # Extract local observations for this agent's units
                local_obs, unit_positions = extract_local_observations(grid, agent_id)
                num_units = len(unit_positions)
                
                if num_units == 0:
                    # Get value estimate even if no units
                    if self.shared_weights:
                        critic_net = self.critic_networks
                    else:
                        critic_net = self.critic_networks[agent_id]
                    value = critic_net(obs_tensor)
                    values[agent_id] = value.squeeze().cpu().item()
                    continue
                
                # Get agent-specific networks
                if self.shared_weights:
                    actor_net = self.actor_networks
                    critic_net = self.critic_networks
                else:
                    actor_net = self.actor_networks[agent_id]
                    critic_net = self.critic_networks[agent_id]
                
                # Extract action masks for each unit
                agent_action_mask = action_mask[agent_id]  # (H, W, 5)
                unit_action_masks = []
                for y, x in unit_positions:
                    unit_action_masks.append(agent_action_mask[y, x])  # (5,)
                unit_action_masks = np.array(unit_action_masks)  # (num_units, 5)
                
                # Convert to tensor: (num_units, 7, 7, 6) -> add batch dimension -> (1, num_units, 7, 7, 6)
                local_obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)  # (1, num_units, 7, 7, 6)
                unit_action_mask_tensor = torch.BoolTensor(unit_action_masks).unsqueeze(0).to(self.device)  # (1, num_units, 5)
                
                # Get action distribution for all units at once
                dist, logits = actor_net(local_obs_tensor, unit_action_mask_tensor)
                
                # Sample actions for all units
                # Set seed for deterministic sampling if provided
                if seed is not None:
                    # Create a generator with the seed for this agent and unit
                    # Combine seed with agent_id and unit index for uniqueness
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(seed + agent_id * 1000)
                    action_samples = dist.sample(generator=generator)  # (num_units,)
                else:
                    action_samples = dist.sample()  # (num_units,)
                
                # Get log probabilities
                log_prob_samples = dist.log_prob(action_samples)  # (num_units,)
                
                # Map actions back to grid positions
                action_samples_np = action_samples.cpu().numpy()
                log_prob_samples_np = log_prob_samples.cpu().numpy()
                for unit_idx, (y, x) in enumerate(unit_positions):
                    actions[agent_id, y, x] = action_samples_np[unit_idx]
                    log_probs[agent_id, y, x] = log_prob_samples_np[unit_idx]
                
                # Get value estimate
                value = critic_net(obs_tensor)
                values[agent_id] = value.squeeze().cpu().item()
        
        return actions, log_probs, values
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: (T,)
            values: (T,)
            dones: (T,)
            next_value: scalar value for next state
            
        Returns:
            advantages: (T,)
            returns: (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        last_gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update_agent(
        self,
        agent_id: int,
        episode_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Update actor and critic for a specific agent."""
        obs = episode_data['obs']
        actions = episode_data['actions']
        old_log_probs = episode_data['log_probs']
        advantages = episode_data['advantages']
        returns = episode_data['returns']
        action_masks = episode_data['action_masks']
        
        # Get agent-specific networks
        if self.shared_weights:
            actor_net = self.actor_networks
            critic_net = self.critic_networks
            actor_optim = self.actor_optimizers
            critic_optim = self.critic_optimizers
        else:
            actor_net = self.actor_networks[agent_id]
            critic_net = self.critic_networks[agent_id]
            actor_optim = self.actor_optimizers[agent_id]
            critic_optim = self.critic_optimizers[agent_id]
        
        # Normalize advantages (handle case where all advantages are the same)
        if advantages.numel() > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std
            else:
                advantages = advantages - advantages.mean()
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Pre-convert to numpy once (outside epoch loop for efficiency)
        obs_np = obs.cpu().numpy()  # (T, 3, H, W)
        # Actions, log_probs, and action_masks may have shape (T, 1, H, W) - squeeze agent dimension
        actions_np = actions.cpu().numpy()  # (T, 1, H, W) or (T, H, W)
        if actions_np.ndim == 4 and actions_np.shape[1] == 1:
            actions_np = actions_np.squeeze(1)  # (T, H, W)
        old_log_probs_np = old_log_probs.cpu().numpy()  # (T, 1, H, W) or (T, H, W)
        if old_log_probs_np.ndim == 4 and old_log_probs_np.shape[1] == 1:
            old_log_probs_np = old_log_probs_np.squeeze(1)  # (T, H, W)
        advantages_np = advantages.cpu().numpy()  # (T,)
        action_masks_np = action_masks.cpu().numpy()  # (T, 1, H, W, 5) or (T, H, W, 5)
        if action_masks_np.ndim == 5 and action_masks_np.shape[1] == 1:
            action_masks_np = action_masks_np.squeeze(1)  # (T, H, W, 5)
        
        # Update for multiple epochs
        for epoch in range(self.update_epochs):
            # Process local observations for this agent across all timesteps
            all_unit_obs_list = []
            all_unit_actions_list = []
            all_unit_old_log_probs_list = []
            all_unit_action_masks_list = []
            all_unit_advantages_list = []
            
            T = obs_np.shape[0]
            
            # First pass: collect all data
            for t in range(T):
                grid_t = obs_np[t]  # (3, H, W)
                
                # Extract local observations for this timestep
                local_obs_t, unit_positions_t = extract_local_observations(grid_t, agent_id)
                num_units_t = len(unit_positions_t)
                
                if num_units_t == 0:
                    continue
                
                # Use vectorized indexing instead of loops
                unit_positions_array = np.array(unit_positions_t)  # (num_units, 2)
                y_coords = unit_positions_array[:, 0]
                x_coords = unit_positions_array[:, 1]
                
                # Extract actions and log probs using vectorized indexing
                unit_actions_t = actions_np[t, y_coords, x_coords]  # (num_units,)
                unit_old_log_probs_t = old_log_probs_np[t, y_coords, x_coords]  # (num_units,)
                unit_action_masks_t = action_masks_np[t, y_coords, x_coords]  # (num_units, 5)
                
                all_unit_obs_list.append(local_obs_t)
                all_unit_actions_list.append(unit_actions_t)
                all_unit_old_log_probs_list.append(unit_old_log_probs_t)
                all_unit_action_masks_list.append(unit_action_masks_t)
                # Expand advantages to match number of units
                all_unit_advantages_list.append(np.full(num_units_t, advantages_np[t], dtype=np.float32))
            
            if len(all_unit_obs_list) == 0:
                # No units for this agent, skip
                actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                entropy = torch.tensor(0.0, device=self.device)
                critic_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # Concatenate all unit observations: (total_units, 7, 7, 6)
                all_unit_obs = np.concatenate(all_unit_obs_list, axis=0)
                all_unit_actions = np.concatenate(all_unit_actions_list, axis=0)
                all_unit_old_log_probs = np.concatenate(all_unit_old_log_probs_list, axis=0)
                all_unit_action_masks = np.concatenate(all_unit_action_masks_list, axis=0)
                all_unit_advantages = np.concatenate(all_unit_advantages_list, axis=0)
                
                # Convert to tensors in one go (faster)
                all_unit_obs_tensor = torch.FloatTensor(all_unit_obs).unsqueeze(0).to(self.device)  # (1, total_units, 7, 7, 6)
                all_unit_action_masks_tensor = torch.BoolTensor(all_unit_action_masks).unsqueeze(0).to(self.device)  # (1, total_units, 5)
                all_unit_actions_tensor = torch.LongTensor(all_unit_actions).to(self.device)  # (total_units,)
                all_unit_old_log_probs_tensor = torch.FloatTensor(all_unit_old_log_probs).to(self.device)  # (total_units,)
                all_unit_advantages_tensor = torch.FloatTensor(all_unit_advantages).to(self.device)  # (total_units,)
                
                # Get current policy
                dist, logits = actor_net(all_unit_obs_tensor, all_unit_action_masks_tensor)
                
                # Compute log probabilities
                log_probs = dist.log_prob(all_unit_actions_tensor)  # (total_units,)
                
                # Compute ratios
                log_prob_diff = log_probs - all_unit_old_log_probs_tensor
                log_prob_diff = torch.clamp(log_prob_diff, min=-10, max=10)
                ratio = torch.exp(log_prob_diff)
                
                # Clamp advantages
                all_unit_advantages_clamped = torch.clamp(all_unit_advantages_tensor, min=-10, max=10)
                
                # PPO clipped objective
                surr1 = ratio * all_unit_advantages_clamped
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * all_unit_advantages_clamped
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                entropy = dist.entropy().mean()
                
                # Critic loss (still uses global observations)
                values = critic_net(obs).squeeze(-1)  # (T,)
                returns_expanded = returns.unsqueeze(-1).expand(T, 1)
                critic_loss = nn.MSELoss()(values.unsqueeze(-1), returns_expanded)
                
                # Total loss
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update actor
                actor_optim.zero_grad()
                critic_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_net.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(critic_net.parameters(), self.max_grad_norm)
                actor_optim.step()
                critic_optim.step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
        
        return {
            'actor_loss': total_actor_loss / self.update_epochs,
            'critic_loss': total_critic_loss / self.update_epochs,
            'entropy': total_entropy / self.update_epochs
        }
    
    def update(self) -> List[Dict[str, float]]:
        """Update all agents using their episode buffers."""
        updates = []
        
        for agent_id in range(self.env.num_agents):
            buffer = self.episode_buffers[agent_id]
            
            if len(buffer.obs) == 0:
                continue
            
            episode_data = buffer.get(self.device)
            
            # Compute GAE
            # Each buffer stores only this agent's data, so shape is (T, 1)
            rewards = episode_data['rewards'].squeeze(-1).cpu().numpy()  # (T,)
            values = episode_data['values'].squeeze(-1).cpu().numpy()  # (T,)
            dones = episode_data['dones'].cpu().numpy()
            
            # Get next value estimate
            obs_tensor = episode_data['obs'][-1:].to(self.device)
            if self.shared_weights:
                critic_net = self.critic_networks
            else:
                critic_net = self.critic_networks[agent_id]
            
            with torch.no_grad():
                next_value = critic_net(obs_tensor).squeeze().cpu().item()
            
            advantages, returns = self.compute_gae(rewards, values, dones, next_value)
            
            # Add to episode data
            episode_data['advantages'] = torch.FloatTensor(advantages).to(self.device)
            episode_data['returns'] = torch.FloatTensor(returns).to(self.device)
            
            # Update agent
            stats = self.update_agent(agent_id, episode_data)
            updates.append(stats)
            
            # Reset buffer
            buffer.reset()
        
        return updates
    
    def train_step(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        done: bool,
        log_prob: np.ndarray,
        value: np.ndarray
    ):
        """Add step to episode buffers."""
        for agent_id in range(self.env.num_agents):
            self.episode_buffers[agent_id].add(
                obs=obs['grid'],
                action=action[agent_id:agent_id+1],
                reward=reward[agent_id:agent_id+1],
                value=value[agent_id:agent_id+1],
                log_prob=log_prob[agent_id:agent_id+1],
                done=done,
                action_mask=obs['action_mask'][agent_id:agent_id+1]
            )
        
        # Update if episode is done
        if done:
            return self.update()
        return None
    
    def save(self, path: str):
        """Save model weights."""
        save_dict = {
            'actor_networks': self.actor_networks.state_dict(),
            'critic_networks': self.critic_networks.state_dict(),
            'shared_weights': self.shared_weights
        }
        
        if self.shared_weights:
            save_dict['actor_optimizer'] = self.actor_optimizers.state_dict()
            save_dict['critic_optimizer'] = self.critic_optimizers.state_dict()
        else:
            save_dict['actor_optimizers'] = [
                opt.state_dict() for opt in self.actor_optimizers
            ]
            save_dict['critic_optimizers'] = [
                opt.state_dict() for opt in self.critic_optimizers
            ]
        
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_networks.load_state_dict(checkpoint['actor_networks'])
        self.critic_networks.load_state_dict(checkpoint['critic_networks'])
        
        if self.shared_weights:
            self.actor_optimizers.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizers.load_state_dict(checkpoint['critic_optimizer'])
        else:
            for i, opt in enumerate(self.actor_optimizers):
                opt.load_state_dict(checkpoint['actor_optimizers'][i])
            for i, opt in enumerate(self.critic_optimizers):
                opt.load_state_dict(checkpoint['critic_optimizers'][i])

