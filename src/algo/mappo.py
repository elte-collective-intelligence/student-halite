"""Multi-Agent Proximal Policy Optimization (MAPPO) for multi-agent Halite."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict

from src.algo.networks import CriticNetwork, LocalActorNetwork
from src.algo.replay_buffer import EpisodeBuffer
from src.env.env import Halite
from src.training.training_util import extract_local_observations


class MAPPO:
    """MAPPO with centralized value function and decentralized actors."""
    
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
        shared_actor: bool = False,
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
        self.shared_actor = shared_actor
        self.device = device
        
        grid_size = env.grid_size
        num_agents = env.num_agents
        
        # Decentralized actors using local 7x7 patches
        if shared_actor:
            self.actor_networks = LocalActorNetwork(
                input_channels=6,  # 6 channels: is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask
                patch_size=7,
                num_actions=5
            ).to(device)
        else:
            self.actor_networks = nn.ModuleList([
                LocalActorNetwork(
                    input_channels=6,
                    patch_size=7,
                    num_actions=5
                ).to(device) for _ in range(num_agents)
            ])
        
        # Centralized critic (sees global state and outputs value for each agent)
        # Critic always uses global observations
        self.critic_network = CriticNetwork(
            input_channels=3,
            grid_size=grid_size,
            centralized=True,
            num_agents=num_agents
        ).to(device)
        
        # Optimizers
        if shared_actor:
            self.actor_optimizer = optim.Adam(self.actor_networks.parameters(), lr=lr_actor)
        else:
            self.actor_optimizers = [
                optim.Adam(net.parameters(), lr=lr_actor) for net in self.actor_networks
            ]
        
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=lr_critic)
        
        # Single episode buffer for all agents (centralized)
        self.episode_buffer = EpisodeBuffer()
        
        self.training_stats = defaultdict(list)
    
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        local_obs_dict: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions using decentralized actor networks with local observations.
        
        Args:
            obs: Observation dictionary (used for critic)
            local_obs_dict: Optional dictionary of local observations per agent.
                          If None, will be generated automatically.
                          {agent_id: {'local_obs': (num_units, 7, 7, 6), 
                                     'unit_positions': [(y, x), ...],
                                     'action_mask': (num_units, 5)}}
            seed: Optional random seed for deterministic action sampling
            
        Returns:
            actions: (num_agents, height, width)
            log_probs: (num_agents, height, width) - log probabilities
            values: (num_agents,) - estimated values from centralized critic
        """
        grid = obs['grid']  # (3, H, W)
        
        # Generate local_obs_dict if not provided
        if local_obs_dict is None:
            local_obs_dict = {}
            action_mask = obs.get('action_mask')
            if action_mask is None:
                # Create default action mask if not provided
                H, W = grid.shape[1], grid.shape[2]
                action_mask = np.ones((self.env.num_agents, H, W, 5), dtype=bool)
            
            for agent_id in range(self.env.num_agents):
                local_obs, unit_positions = extract_local_observations(grid, agent_id)
                agent_action_mask = action_mask[agent_id]  # (H, W, 5)
                unit_action_masks = []
                for y, x in unit_positions:
                    unit_action_masks.append(agent_action_mask[y, x])
                unit_action_masks = np.array(unit_action_masks) if unit_action_masks else np.zeros((0, 5), dtype=bool)
                
                local_obs_dict[agent_id] = {
                    'local_obs': local_obs,
                    'unit_positions': unit_positions,
                    'action_mask': unit_action_masks
                }
        
        # Get actual grid dimensions
        H, W = grid.shape[1], grid.shape[2]
        
        actions = np.zeros((self.env.num_agents, H, W), dtype=np.int32)
        log_probs = np.zeros((self.env.num_agents, H, W), dtype=np.float32)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(grid).unsqueeze(0).to(self.device)
            
            # Get centralized value estimates (always uses global observations)
            values_tensor = self.critic_network(obs_tensor)
            values = values_tensor.squeeze(0).cpu().numpy()  # (num_agents,)
            
            # Batch process all agents if using shared actor (faster)
            if self.shared_actor:
                # Collect all units from all agents
                all_agent_local_obs = []
                all_agent_unit_positions = []
                all_agent_unit_masks = []
                
                for agent_id in range(self.env.num_agents):
                    agent_local_data = local_obs_dict.get(agent_id, {})
                    local_obs = agent_local_data.get('local_obs', np.zeros((0, 7, 7, 6), dtype=np.float32))
                    unit_positions = agent_local_data.get('unit_positions', [])
                    unit_action_masks = agent_local_data.get('action_mask', np.zeros((0, 5), dtype=bool))
                    
                    num_units = len(unit_positions)
                    if num_units > 0:
                        all_agent_local_obs.append(local_obs)
                        all_agent_unit_positions.append((agent_id, unit_positions))
                        all_agent_unit_masks.append(unit_action_masks)
                
                if len(all_agent_local_obs) > 0:
                    # Concatenate all units: (total_units, 7, 7, 6)
                    all_local_obs = np.concatenate(all_agent_local_obs, axis=0)
                    all_unit_masks = np.concatenate(all_agent_unit_masks, axis=0)
                    
                    # Convert to tensor and process in one batch
                    local_obs_tensor = torch.FloatTensor(all_local_obs).unsqueeze(0).to(self.device)  # (1, total_units, 7, 7, 6)
                    unit_mask_tensor = torch.BoolTensor(all_unit_masks).unsqueeze(0).to(self.device)  # (1, total_units, 5)
                    
                    # Get action distribution for all units at once
                    dist, logits = self.actor_networks(local_obs_tensor, unit_mask_tensor)
                    
                    # Sample actions for all units
                    action_samples = dist.sample()  # (total_units,)
                    
                    # Get log probabilities
                    log_prob_samples = dist.log_prob(action_samples)  # (total_units,)
                    
                    # Map actions back to grid positions using vectorized indexing
                    action_samples_np = action_samples.cpu().numpy()
                    log_prob_samples_np = log_prob_samples.cpu().numpy()
                    
                    unit_idx = 0
                    for agent_id, unit_positions in all_agent_unit_positions:
                        for y, x in unit_positions:
                            actions[agent_id, y, x] = action_samples_np[unit_idx]
                            log_probs[agent_id, y, x] = log_prob_samples_np[unit_idx]
                            unit_idx += 1
            else:
                # Process each agent separately (independent actors)
                for agent_id in range(self.env.num_agents):
                    # Get agent-specific actor network
                    actor_net = self.actor_networks[agent_id]
                    
                    # Process local observations per unit
                    agent_local_data = local_obs_dict.get(agent_id, {})
                    local_obs = agent_local_data.get('local_obs', np.zeros((0, 7, 7, 6), dtype=np.float32))
                    unit_positions = agent_local_data.get('unit_positions', [])
                    unit_action_masks = agent_local_data.get('action_mask', np.zeros((0, 5), dtype=bool))
                    
                    num_units = len(unit_positions)
                    if num_units == 0:
                        continue
                    
                    # Convert to tensor: (num_units, 7, 7, 6) -> add batch dimension -> (1, num_units, 7, 7, 6)
                    local_obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)  # (1, num_units, 7, 7, 6)
                    unit_action_mask_tensor = torch.BoolTensor(unit_action_masks).unsqueeze(0).to(self.device)  # (1, num_units, 5)
                    
                    # Get action distribution for all units at once
                    dist, logits = actor_net(local_obs_tensor, unit_action_mask_tensor)
                    
                    # Sample actions for all units
                    # Set seed for deterministic sampling if provided
                    if seed is not None:
                        # Create a generator with the seed for this agent and unit
                        # Combine seed with agent_id for uniqueness
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(seed + agent_id * 1000)
                        action_samples = dist.sample(generator=generator)  # (num_units,)
                    else:
                        action_samples = dist.sample()  # (num_units,)
                    
                    # Get log probabilities
                    log_prob_samples = dist.log_prob(action_samples)  # (num_units,)
                    
                    # Map actions back to grid positions using vectorized indexing
                    action_samples_np = action_samples.cpu().numpy()
                    log_prob_samples_np = log_prob_samples.cpu().numpy()
                    for unit_idx, (y, x) in enumerate(unit_positions):
                        actions[agent_id, y, x] = action_samples_np[unit_idx]
                        log_probs[agent_id, y, x] = log_prob_samples_np[unit_idx]
        
        return actions, log_probs, values
    
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation for all agents.
        
        Args:
            rewards: (T, num_agents)
            values: (T, num_agents)
            dones: (T,)
            next_values: (num_agents,) - values for next state
            
        Returns:
            advantages: (T, num_agents)
            returns: (T, num_agents)
        """
        T, num_agents = rewards.shape
        advantages = np.zeros((T, num_agents), dtype=np.float32)
        returns = np.zeros((T, num_agents), dtype=np.float32)
        
        for agent_id in range(num_agents):
            last_gae = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = next_values[agent_id]
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value_t = values[t + 1, agent_id]
                
                delta = rewards[t, agent_id] + self.gamma * next_value_t * next_non_terminal - values[t, agent_id]
                advantages[t, agent_id] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                returns[t, agent_id] = advantages[t, agent_id] + values[t, agent_id]
        
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """Update actors and centralized critic using episode buffer."""
        if len(self.episode_buffer.obs) == 0:
            return {}
        
        episode_data = self.episode_buffer.get(self.device)
        
        obs = episode_data['obs']
        actions = episode_data['actions']  # (T, num_agents, H, W)
        old_log_probs = episode_data['log_probs']  # (T, num_agents, H, W)
        rewards = episode_data['rewards'].cpu().numpy()  # (T, num_agents)
        values = episode_data['values'].cpu().numpy()  # (T, num_agents)
        dones = episode_data['dones'].cpu().numpy()  # (T,)
        action_masks = episode_data['action_masks']  # (T, num_agents, H, W, 5)
        
        # Get next value estimate from centralized critic
        with torch.no_grad():
            next_obs = obs[-1:].to(self.device)
            next_values_tensor = self.critic_network(next_obs)
            next_values = next_values_tensor.squeeze(0).cpu().numpy()  # (num_agents,)
        
        # Compute GAE for all agents
        advantages, returns = self.compute_gae(rewards, values, dones, next_values)
        
        # Convert to tensors
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)  # (T, num_agents)
        returns_tensor = torch.FloatTensor(returns).to(self.device)  # (T, num_agents)
        
        # Normalize advantages per agent (handle case where all advantages are the same)
        advantages_mean = advantages_tensor.mean(dim=0, keepdim=True)
        advantages_std = advantages_tensor.std(dim=0, keepdim=True)
        # Only normalize if std is sufficiently large
        advantages_tensor = torch.where(
            advantages_std > 1e-8,
            (advantages_tensor - advantages_mean) / advantages_std,
            advantages_tensor - advantages_mean
        )
        
        T, num_agents, H, W = actions.shape
        
        # Pre-convert to numpy once (outside epoch loop for efficiency)
        obs_np = obs.cpu().numpy()  # (T, 3, H, W)
        actions_np = actions.cpu().numpy()  # (T, num_agents, H, W)
        old_log_probs_np = old_log_probs.cpu().numpy()  # (T, num_agents, H, W)
        advantages_np = advantages_tensor.cpu().numpy()  # (T, num_agents)
        action_masks_np = action_masks.cpu().numpy()  # (T, num_agents, H, W, 5)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Update for multiple epochs
        for epoch in range(self.update_epochs):
            # Update centralized critic
            values_pred = self.critic_network(obs)  # (T, num_agents)
            # Clamp values to prevent extreme losses
            values_pred_clamped = torch.clamp(values_pred, min=-100, max=100)
            returns_clamped = torch.clamp(returns_tensor, min=-100, max=100)
            critic_loss = nn.MSELoss()(values_pred_clamped, returns_clamped)
            
            # Check for NaN or Inf
            if torch.isnan(critic_loss) or torch.isinf(critic_loss):
                critic_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # Update decentralized actors using local observations
            actor_losses = []
            entropies = []
            
            for agent_id in range(num_agents):
                # Get agent-specific actor
                if self.shared_actor:
                    actor_net = self.actor_networks
                else:
                    actor_net = self.actor_networks[agent_id]
                
                # Process local observations for this agent across all timesteps
                # Pre-allocate lists (faster than extending)
                all_unit_obs_list = []
                all_unit_actions_list = []
                all_unit_old_log_probs_list = []
                all_unit_action_masks_list = []
                all_unit_advantages_list = []
                
                agent_actions = actions_np[:, agent_id]  # (T, H, W)
                agent_old_log_probs = old_log_probs_np[:, agent_id]  # (T, H, W)
                agent_advantages = advantages_np[:, agent_id]  # (T,)
                agent_action_masks = action_masks_np[:, agent_id]  # (T, H, W, 5)
                
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
                    unit_actions_t = agent_actions[t, y_coords, x_coords]  # (num_units,)
                    unit_old_log_probs_t = agent_old_log_probs[t, y_coords, x_coords]  # (num_units,)
                    unit_action_masks_t = agent_action_masks[t, y_coords, x_coords]  # (num_units, 5)
                    
                    all_unit_obs_list.append(local_obs_t)
                    all_unit_actions_list.append(unit_actions_t)
                    all_unit_old_log_probs_list.append(unit_old_log_probs_t)
                    all_unit_action_masks_list.append(unit_action_masks_t)
                    # Expand advantages to match number of units
                    all_unit_advantages_list.append(np.full(num_units_t, agent_advantages[t], dtype=np.float32))
                
                if len(all_unit_obs_list) == 0:
                    # No units for this agent, skip
                    actor_losses.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                    entropies.append(torch.tensor(0.0, device=self.device))
                    continue
                
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
                agent_actor_loss = -torch.min(surr1, surr2).mean()
                
                # Check for NaN or Inf
                if torch.isnan(agent_actor_loss) or torch.isinf(agent_actor_loss):
                    agent_actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                actor_losses.append(agent_actor_loss)
                
                # Entropy
                entropy = dist.entropy().mean()
                if torch.isnan(entropy) or torch.isinf(entropy):
                    entropy = torch.tensor(0.0, device=self.device)
                entropies.append(entropy)
            
            # Average actor loss across agents
            actor_loss = torch.stack(actor_losses).mean()
            entropy = torch.stack(entropies).mean()
            
            # Check for NaN before computing total loss
            if torch.isnan(actor_loss) or torch.isinf(actor_loss):
                actor_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(entropy) or torch.isinf(entropy):
                entropy = torch.tensor(0.0, device=self.device)
            
            # Total loss
            total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Skip update if loss is invalid
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)) and total_loss.item() < 1e10:
                # Update networks
                if self.shared_actor:
                    self.actor_optimizer.zero_grad()
                else:
                    for opt in self.actor_optimizers:
                        opt.zero_grad()
                
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.shared_actor:
                    torch.nn.utils.clip_grad_norm_(self.actor_networks.parameters(), self.max_grad_norm)
                else:
                    for actor_net in self.actor_networks:
                        torch.nn.utils.clip_grad_norm_(actor_net.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), self.max_grad_norm)
                
                if self.shared_actor:
                    self.actor_optimizer.step()
                else:
                    for opt in self.actor_optimizers:
                        opt.step()
                
                self.critic_optimizer.step()
            
            # Accumulate losses (handle NaN/Inf safely)
            actor_loss_val = actor_loss.item() if not (torch.isnan(actor_loss) or torch.isinf(actor_loss)) else 0.0
            critic_loss_val = critic_loss.item() if not (torch.isnan(critic_loss) or torch.isinf(critic_loss)) else 0.0
            entropy_val = entropy.item() if not (torch.isnan(entropy) or torch.isinf(entropy)) else 0.0
            
            total_actor_loss += actor_loss_val
            total_critic_loss += critic_loss_val
            total_entropy += entropy_val
        
        # Reset buffer
        self.episode_buffer.reset()
        
        return {
            'actor_loss': total_actor_loss / self.update_epochs,
            'critic_loss': total_critic_loss / self.update_epochs,
            'entropy': total_entropy / self.update_epochs
        }
    
    def train_step(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        done: bool,
        log_prob: np.ndarray,
        value: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """Add step to episode buffer."""
        self.episode_buffer.add(
            obs=obs['grid'],
            action=action,
            reward=reward,
            value=value,
            log_prob=log_prob,
            done=done,
            action_mask=obs['action_mask']
        )
        
        # Update if episode is done
        if done:
            return self.update()
        return None
    
    def save(self, path: str):
        """Save model weights."""
        save_dict = {
            'actor_networks': self.actor_networks.state_dict(),
            'critic_network': self.critic_network.state_dict(),
            'shared_actor': self.shared_actor
        }
        
        if self.shared_actor:
            save_dict['actor_optimizer'] = self.actor_optimizer.state_dict()
        else:
            save_dict['actor_optimizers'] = [
                opt.state_dict() for opt in self.actor_optimizers
            ]
        
        save_dict['critic_optimizer'] = self.critic_optimizer.state_dict()
        
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_networks.load_state_dict(checkpoint['actor_networks'])
        self.critic_network.load_state_dict(checkpoint['critic_network'])
        
        if self.shared_actor:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        else:
            for i, opt in enumerate(self.actor_optimizers):
                opt.load_state_dict(checkpoint['actor_optimizers'][i])
        
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

