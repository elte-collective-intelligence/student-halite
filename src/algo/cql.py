"""Centralized Q-Learning algorithm for multi-agent Halite."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict

from src.algo.networks import LocalCentralizedQNetwork
from src.algo.replay_buffer import ReplayBuffer
from src.env.env import Halite
from src.training.training_util import extract_local_observations


class CentralizedQLearning:
    """Centralized Q-Learning with joint-state centralized critic."""
    
    def __init__(
        self,
        env: Halite,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 100,
        device: str = 'cpu'
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        grid_size = env.grid_size
        num_agents = env.num_agents
        
        # Initialize networks (using local 7x7 observations)
        self.q_network = LocalCentralizedQNetwork(
            input_channels=6,  # 6 channels: is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask
            patch_size=7,
            num_agents=num_agents,
            num_actions=5
        ).to(device)
        
        self.target_q_network = LocalCentralizedQNetwork(
            input_channels=6,
            patch_size=7,
            num_agents=num_agents,
            num_actions=5
        ).to(device)
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            num_agents=num_agents,
            grid_size=grid_size,
            num_actions=5
        )
        
        self.step_count = 0
        self.training_stats = defaultdict(list)
    
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Select actions using epsilon-greedy policy.
        
        Args:
            obs: Observation dictionary with 'grid', 'action_mask', 'step_count'
            seed: Optional random seed for deterministic action sampling
            
        Returns:
            actions: (num_agents, height, width)
        """
        grid = obs['grid']  # (3, H, W)
        action_mask = obs['action_mask']  # (num_agents, H, W, 5)
        
        # Get actual grid dimensions
        H, W = grid.shape[1], grid.shape[2]
        
        actions = np.zeros((self.env.num_agents, H, W), dtype=np.int32)
        
        # Collect all units from all agents
        all_agent_local_obs = []
        all_agent_unit_positions = []
        all_agent_unit_masks = []
        
        for agent_id in range(self.env.num_agents):
            local_obs, unit_positions = extract_local_observations(grid, agent_id)
            num_units = len(unit_positions)
            
            if num_units == 0:
                continue
            
            # Extract action masks for each unit
            agent_action_mask = action_mask[agent_id]  # (H, W, 5)
            unit_action_masks = []
            for y, x in unit_positions:
                unit_action_masks.append(agent_action_mask[y, x])  # (5,)
            unit_action_masks = np.array(unit_action_masks)  # (num_units, 5)
            
            all_agent_local_obs.append(local_obs)
            all_agent_unit_positions.append((agent_id, unit_positions))
            all_agent_unit_masks.append(unit_action_masks)
        
        if len(all_agent_local_obs) == 0:
            return actions
        
        # Create RNG with seed if provided
        if seed is not None:
            rng = np.random.default_rng(seed)
            use_rng = True
        else:
            rng = None
            use_rng = False
        
        # Epsilon-greedy exploration
        if (rng.random() if use_rng else np.random.random()) < self.epsilon:
            # Random actions for all units
            for agent_id, unit_positions in all_agent_unit_positions:
                for y, x in unit_positions:
                    actions[agent_id, y, x] = rng.integers(0, 5) if use_rng else np.random.randint(0, 5)
        else:
            # Greedy actions using local observations
            with torch.no_grad():
                # Concatenate all units: (total_units, 7, 7, 6)
                all_local_obs = np.concatenate(all_agent_local_obs, axis=0)
                all_unit_masks = np.concatenate(all_agent_unit_masks, axis=0)
                
                # Convert to tensor
                local_obs_tensor = torch.FloatTensor(all_local_obs).unsqueeze(0).to(self.device)  # (1, total_units, 7, 7, 6)
                unit_mask_tensor = torch.BoolTensor(all_unit_masks).unsqueeze(0).to(self.device)  # (1, total_units, 5)
                
                # Reshape action mask to (1, num_agents, total_units, 5) for centralized network
                # We need to organize by agent
                agent_unit_indices = {}
                unit_idx = 0
                for agent_id, unit_positions in all_agent_unit_positions:
                    num_units_agent = len(unit_positions)
                    agent_unit_indices[agent_id] = (unit_idx, unit_idx + num_units_agent)
                    unit_idx += num_units_agent
                
                # Create action mask organized by agent: (1, num_agents, total_units, 5)
                centralized_action_mask = torch.zeros((1, self.env.num_agents, all_unit_masks.shape[0], 5), dtype=torch.bool, device=self.device)
                for agent_id, (start_idx, end_idx) in agent_unit_indices.items():
                    centralized_action_mask[0, agent_id, start_idx:end_idx, :] = unit_mask_tensor[0, start_idx:end_idx, :]
                
                q_values = self.q_network(local_obs_tensor, centralized_action_mask)
                # q_values: (1, num_agents, total_units, 5)
                
                # Select best action for each unit
                q_values = q_values.squeeze(0)  # (num_agents, total_units, 5)
                unit_idx = 0
                for agent_id, unit_positions in all_agent_unit_positions:
                    num_units_agent = len(unit_positions)
                    agent_q_values = q_values[agent_id, unit_idx:unit_idx + num_units_agent]  # (num_units_agent, 5)
                    agent_actions = agent_q_values.argmax(dim=-1).cpu().numpy()  # (num_units_agent,)
                    
                    for unit_local_idx, (y, x) in enumerate(unit_positions):
                        actions[agent_id, y, x] = agent_actions[unit_local_idx]
                    
                    unit_idx += num_units_agent
        
        return actions
    
    def update(self) -> Optional[Dict[str, float]]:
        """Perform one update step using a batch from the replay buffer with local observations."""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        batch = self.replay_buffer.sample(self.batch_size, self.device)
        
        # Extract local observations from batch
        obs = batch['obs']
        obs_shape = obs.shape
        
        # Fix obs shape if needed
        if len(obs_shape) == 5:
            if obs_shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs_shape[0] == obs_shape[1]:
                obs = obs[:, 0, :, :, :]
        
        obs_np = obs.cpu().numpy()  # (batch, 3, H, W)
        batch_size = obs_np.shape[0]
        
        # Collect all unit observations and actions across batch and agents
        all_unit_obs_list = []
        all_unit_actions_list = []
        all_unit_action_masks_list = []
        all_unit_rewards_list = []
        all_unit_dones_list = []
        unit_agent_ids = []  # Track which agent each unit belongs to
        batch_indices = []  # Track which batch item each unit belongs to
        
        actions = batch['action']
        if actions.dim() == 4:
            if actions.size(1) == 1:
                actions = actions.squeeze(1)
        actions_np = actions.cpu().numpy()  # (batch, num_agents, H, W)
        
        action_mask = batch['action_mask']
        if action_mask is not None:
            if len(action_mask.shape) == 5:
                if action_mask.size(1) == 1:
                    action_mask = action_mask.squeeze(1)
        action_mask_np = action_mask.cpu().numpy() if action_mask is not None else None
        
        # Extract local observations for each batch item and agent
        for b in range(batch_size):
            grid_b = obs_np[b]  # (3, H, W)
            
            for agent_id in range(self.env.num_agents):
                local_obs_b, unit_positions_b = extract_local_observations(grid_b, agent_id)
                num_units_b = len(unit_positions_b)
                
                if num_units_b == 0:
                    continue
                
                # Extract actions and action masks for units
                unit_positions_array = np.array(unit_positions_b)
                y_coords = unit_positions_array[:, 0]
                x_coords = unit_positions_array[:, 1]
                
                unit_actions_b = actions_np[b, agent_id, y_coords, x_coords]  # (num_units,)
                
                unit_action_masks_b = None
                if action_mask_np is not None:
                    unit_action_masks_b = action_mask_np[b, agent_id, y_coords, x_coords]  # (num_units, 5)
                
                all_unit_obs_list.append(local_obs_b)
                all_unit_actions_list.append(unit_actions_b)
                all_unit_action_masks_list.append(unit_action_masks_b)
                unit_agent_ids.extend([agent_id] * num_units_b)
                batch_indices.extend([b] * num_units_b)
        
        if len(all_unit_obs_list) == 0:
            return None
        
        # Concatenate all unit observations
        all_unit_obs = np.concatenate(all_unit_obs_list, axis=0)  # (total_units, 7, 7, 6)
        all_unit_actions = np.concatenate(all_unit_actions_list, axis=0)  # (total_units,)
        all_unit_action_masks = None
        if all_unit_action_masks_list[0] is not None:
            all_unit_action_masks = np.concatenate([m for m in all_unit_action_masks_list if m is not None], axis=0)  # (total_units, 5)
        
        # Convert to tensors
        all_unit_obs_tensor = torch.FloatTensor(all_unit_obs).unsqueeze(0).to(self.device)  # (1, total_units, 7, 7, 6)
        all_unit_actions_tensor = torch.LongTensor(all_unit_actions).to(self.device)  # (total_units,)
        
        # Create centralized action mask: (1, num_agents, total_units, 5)
        centralized_action_mask = torch.zeros((1, self.env.num_agents, all_unit_obs.shape[0], 5), dtype=torch.bool, device=self.device)
        if all_unit_action_masks is not None:
            all_unit_action_masks_tensor = torch.BoolTensor(all_unit_action_masks).to(self.device)  # (total_units, 5)
            for idx, agent_id in enumerate(unit_agent_ids):
                centralized_action_mask[0, agent_id, idx, :] = all_unit_action_masks_tensor[idx, :]
        
        # Current Q-values
        q_values = self.q_network(all_unit_obs_tensor, centralized_action_mask)  # (1, num_agents, total_units, 5)
        q_values = q_values.squeeze(0)  # (num_agents, total_units, 5)
        
        # Gather Q-values for taken actions
        total_units = len(unit_agent_ids)
        q_selected = torch.zeros(total_units, device=self.device)
        for idx, (agent_id, action) in enumerate(zip(unit_agent_ids, all_unit_actions_tensor)):
            q_selected[idx] = q_values[agent_id, idx, action]
        
        # Next Q-values from target network
        with torch.no_grad():
            next_obs = batch['next_obs']
            next_obs_shape = next_obs.shape
            
            if len(next_obs_shape) == 5:
                if next_obs_shape[1] == 1:
                    next_obs = next_obs.squeeze(1)
                elif next_obs_shape[0] == next_obs_shape[1]:
                    next_obs = next_obs[:, 0, :, :, :]
            
            next_obs_np = next_obs.cpu().numpy()  # (batch, 3, H, W)
            
            # Extract local observations for next states
            all_next_unit_obs_list = []
            all_next_unit_action_masks_list = []
            next_unit_agent_ids = []
            next_batch_indices = []
            
            next_action_mask = batch['next_action_mask']
            if next_action_mask is not None:
                if len(next_action_mask.shape) == 5:
                    if next_action_mask.size(1) == 1:
                        next_action_mask = next_action_mask.squeeze(1)
                    elif next_action_mask.shape[0] == next_action_mask.shape[1]:
                        next_action_mask = next_action_mask[:, 0, :, :, :]
            next_action_mask_np = next_action_mask.cpu().numpy() if next_action_mask is not None else None
            
            for b in range(batch_size):
                next_grid_b = next_obs_np[b]  # (3, H, W)
                
                for agent_id in range(self.env.num_agents):
                    next_local_obs_b, next_unit_positions_b = extract_local_observations(next_grid_b, agent_id)
                    next_num_units_b = len(next_unit_positions_b)
                    
                    if next_num_units_b == 0:
                        continue
                    
                    next_unit_action_masks_b = None
                    if next_action_mask_np is not None:
                        next_unit_positions_array = np.array(next_unit_positions_b)
                        next_y_coords = next_unit_positions_array[:, 0]
                        next_x_coords = next_unit_positions_array[:, 1]
                        next_unit_action_masks_b = next_action_mask_np[b, agent_id, next_y_coords, next_x_coords]  # (next_num_units, 5)
                    
                    all_next_unit_obs_list.append(next_local_obs_b)
                    all_next_unit_action_masks_list.append(next_unit_action_masks_b)
                    next_unit_agent_ids.extend([agent_id] * next_num_units_b)
                    next_batch_indices.extend([b] * next_num_units_b)
            
            if len(all_next_unit_obs_list) == 0:
                next_q_max_flat = torch.zeros_like(q_selected)
            else:
                all_next_unit_obs = np.concatenate(all_next_unit_obs_list, axis=0)  # (total_next_units, 7, 7, 6)
                all_next_unit_action_masks = None
                if all_next_unit_action_masks_list[0] is not None:
                    all_next_unit_action_masks = np.concatenate([m for m in all_next_unit_action_masks_list if m is not None], axis=0)
                
                all_next_unit_obs_tensor = torch.FloatTensor(all_next_unit_obs).unsqueeze(0).to(self.device)  # (1, total_next_units, 7, 7, 6)
                
                # Create centralized action mask for next state
                next_centralized_action_mask = torch.zeros((1, self.env.num_agents, all_next_unit_obs.shape[0], 5), dtype=torch.bool, device=self.device)
                if all_next_unit_action_masks is not None:
                    all_next_unit_action_masks_tensor = torch.BoolTensor(all_next_unit_action_masks).to(self.device)  # (total_next_units, 5)
                    for idx, agent_id in enumerate(next_unit_agent_ids):
                        next_centralized_action_mask[0, agent_id, idx, :] = all_next_unit_action_masks_tensor[idx, :]
                
                next_q_values = self.target_q_network(all_next_unit_obs_tensor, next_centralized_action_mask)  # (1, num_agents, total_next_units, 5)
                next_q_values = next_q_values.squeeze(0)  # (num_agents, total_next_units, 5)
                
                # Handle -inf from masking
                next_q_values_clamped = next_q_values.clone()
                next_q_values_clamped[next_q_values_clamped == float('-inf')] = -1e6
                next_q_max = next_q_values_clamped.max(dim=-1)[0]  # (num_agents, total_next_units)
                
                # Map back to original units (use max Q-value per agent per batch item)
                next_q_max_by_batch_agent = {}
                for idx, (b_idx, agent_id) in enumerate(zip(next_batch_indices, next_unit_agent_ids)):
                    key = (b_idx, agent_id)
                    if key not in next_q_max_by_batch_agent:
                        next_q_max_by_batch_agent[key] = []
                    next_q_max_by_batch_agent[key].append(next_q_max[agent_id, idx].item())
                
                next_q_max_flat = torch.zeros_like(q_selected)
                for idx, (b_idx, agent_id) in enumerate(zip(batch_indices, unit_agent_ids)):
                    key = (b_idx, agent_id)
                    if key in next_q_max_by_batch_agent and len(next_q_max_by_batch_agent[key]) > 0:
                        next_q_max_flat[idx] = max(next_q_max_by_batch_agent[key])
            
            # Target Q-values
            rewards = batch['reward']
            if rewards.dim() == 2:
                rewards = rewards.squeeze(1) if rewards.size(1) == 1 else rewards
            rewards_np = rewards.cpu().numpy()  # (batch, num_agents)
            
            done = batch['done']
            if done.dim() == 2:
                done = done.squeeze(1) if done.size(1) == 1 else done
            done_np = done.cpu().numpy()  # (batch,)
            
            # Expand rewards and done to match units
            rewards_expanded = torch.zeros_like(q_selected)
            done_expanded = torch.zeros_like(q_selected, dtype=torch.float)
            for idx, (b_idx, agent_id) in enumerate(zip(batch_indices, unit_agent_ids)):
                rewards_expanded[idx] = torch.tensor(rewards_np[b_idx, agent_id], dtype=torch.float32, device=self.device)
                done_expanded[idx] = torch.tensor(done_np[b_idx], dtype=torch.float32, device=self.device)
            
            target_q = rewards_expanded + self.gamma * next_q_max_flat * (1 - done_expanded)  # (total_units,)
        
        # Compute loss
        q_selected_clamped = torch.clamp(q_selected, min=-100, max=100)
        target_q_clamped = torch.clamp(target_q, min=-100, max=100)
        loss = nn.MSELoss()(q_selected_clamped, target_q_clamped)
        
        # Check for NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Update network only if loss is valid
        if not (torch.isnan(loss) or torch.isinf(loss)) and loss.item() < 1e10:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Get mean Q-value safely
        q_mean_val = q_selected_clamped.mean().item()
        if np.isnan(q_mean_val) or np.isinf(q_mean_val):
            q_mean_val = 0.0
        
        loss_val = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0
        
        return {
            'loss': loss_val,
            'epsilon': self.epsilon,
            'q_mean': q_mean_val
        }
    
    def train_step(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        done: bool
    ) -> Optional[Dict[str, float]]:
        """Add transition to buffer and update if enough samples."""
        self.replay_buffer.push(
            obs=obs['grid'],
            action=action,
            reward=reward,
            next_obs=next_obs['grid'],
            done=done,
            action_mask=obs['action_mask'],
            next_action_mask=next_obs['action_mask']
        )
        
        return self.update()
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']

