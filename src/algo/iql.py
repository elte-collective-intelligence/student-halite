"""Independent Q-Learning (IQL) algorithm for multi-agent Halite."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Any, Optional, List
from collections import defaultdict

from src.algo.networks import LocalQNetwork
from src.algo.replay_buffer import ReplayBuffer
from src.env.env import Halite
from src.training.training_util import extract_local_observations


class IQL:
    """Independent Q-Learning with per-agent Q-networks and target networks."""
    
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
        shared_network: bool = False,
        device: str = 'cpu'
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.shared_network = shared_network
        self.device = device
        
        grid_size = env.grid_size
        num_agents = env.num_agents
        
        # Initialize per-agent Q-networks (using local 7x7 observations)
        if shared_network:
            # Shared network architecture (weights shared across agents)
            self.q_networks = LocalQNetwork(
                input_channels=6,  # 6 channels: is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask
                patch_size=7,
                num_actions=5
            ).to(device)
            self.target_q_networks = LocalQNetwork(
                input_channels=6,
                patch_size=7,
                num_actions=5
            ).to(device)
        else:
            # Independent networks for each agent
            self.q_networks = nn.ModuleList([
                LocalQNetwork(
                    input_channels=6,
                    patch_size=7,
                    num_actions=5
                ).to(device) for _ in range(num_agents)
            ])
            self.target_q_networks = nn.ModuleList([
                LocalQNetwork(
                    input_channels=6,
                    patch_size=7,
                    num_actions=5
                ).to(device) for _ in range(num_agents)
            ])
        
        # Initialize target networks
        if shared_network:
            self.target_q_networks.load_state_dict(self.q_networks.state_dict())
            self.target_q_networks.eval()
        else:
            for i in range(num_agents):
                self.target_q_networks[i].load_state_dict(self.q_networks[i].state_dict())
                self.target_q_networks[i].eval()
        
        # Optimizers
        if shared_network:
            self.optimizers = optim.Adam(self.q_networks.parameters(), lr=lr)
        else:
            self.optimizers = [
                optim.Adam(net.parameters(), lr=lr) for net in self.q_networks
            ]
        
        # Replay buffer (shared or per-agent)
        self.replay_buffers = [
            ReplayBuffer(
                capacity=buffer_size,
                num_agents=1,  # Each agent has its own buffer
                grid_size=grid_size,
                num_actions=5
            ) for _ in range(num_agents)
        ]
        
        self.step_count = 0
        self.training_stats = defaultdict(list)
    
    def select_action(
        self,
        obs: Dict[str, np.ndarray],
        seed: Optional[int] = None
    ) -> np.ndarray:
        """Select actions using epsilon-greedy policy for each agent independently.
        
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
        
        # Select action for each agent independently
        for agent_id in range(self.env.num_agents):
            agent_ownership = grid[0] == (agent_id + 1)
            
            if not np.any(agent_ownership):
                continue  # Agent has no cells
            
            # Get agent-specific Q-network
            if self.shared_network:
                q_net = self.q_networks
            else:
                q_net = self.q_networks[agent_id]
            
            # Epsilon-greedy exploration
            # Extract local observations for this agent's units
            local_obs, unit_positions = extract_local_observations(grid, agent_id)
            num_units = len(unit_positions)
            
            if num_units == 0:
                continue  # Agent has no cells
            
            # Create RNG with seed if provided
            if seed is not None:
                rng = np.random.default_rng(seed + agent_id * 1000)
                use_rng = True
            else:
                rng = None
                use_rng = False
            
            if (rng.random() if use_rng else np.random.random()) < self.epsilon:
                # Random actions for owned cells
                for y, x in unit_positions:
                    actions[agent_id, y, x] = rng.integers(0, 5) if use_rng else np.random.randint(0, 5)
            else:
                # Greedy actions using local observations
                with torch.no_grad():
                    # Extract action masks for each unit
                    agent_action_mask = action_mask[agent_id]  # (H, W, 5)
                    unit_action_masks = []
                    for y, x in unit_positions:
                        unit_action_masks.append(agent_action_mask[y, x])  # (5,)
                    unit_action_masks = np.array(unit_action_masks)  # (num_units, 5)
                    
                    # Convert to tensor: (num_units, 7, 7, 6) -> add batch dimension -> (1, num_units, 7, 7, 6)
                    local_obs_tensor = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)  # (1, num_units, 7, 7, 6)
                    unit_action_mask_tensor = torch.BoolTensor(unit_action_masks).unsqueeze(0).to(self.device)  # (1, num_units, 5)
                    
                    q_values = q_net(local_obs_tensor, unit_action_mask_tensor)
                    # q_values: (1, num_units, 5)
                    
                    # Select best action for each unit
                    agent_actions = q_values.squeeze(0).argmax(dim=-1).cpu().numpy()  # (num_units,)
                    for unit_idx, (y, x) in enumerate(unit_positions):
                        actions[agent_id, y, x] = agent_actions[unit_idx]
        
        return actions
    
    def update(self, agent_id: int) -> Optional[Dict[str, float]]:
        """Perform one update step for a specific agent using local observations."""
        buffer = self.replay_buffers[agent_id]
        
        if len(buffer) < self.batch_size:
            return None
        
        batch = buffer.sample(self.batch_size, self.device)
        
        # Get agent-specific network
        if self.shared_network:
            q_net = self.q_networks
            target_q_net = self.target_q_networks
            optimizer = self.optimizers
        else:
            q_net = self.q_networks[agent_id]
            target_q_net = self.target_q_networks[agent_id]
            optimizer = self.optimizers[agent_id]
        
        # Extract local observations from batch
        obs = batch['obs']
        obs_shape = obs.shape
        
        # Fix obs shape if it has extra dimension
        if len(obs_shape) == 5:
            if obs_shape[1] == 1:
                obs = obs.squeeze(1)
            elif obs_shape[0] == obs_shape[1]:
                obs = obs[:, 0, :, :, :]
            else:
                print(f"Error: Unexpected 5D obs shape: {obs_shape}")
                return None
        elif len(obs_shape) != 4:
            print(f"Warning: Unexpected obs shape in IQL: {obs_shape}, expected 4 dims")
            return None
        
        # Convert to numpy for local observation extraction
        obs_np = obs.cpu().numpy()  # (batch, 3, H, W)
        batch_size = obs_np.shape[0]
        
        # Collect all unit observations and actions across batch
        all_unit_obs_list = []
        all_unit_actions_list = []
        all_unit_action_masks_list = []
        all_unit_rewards_list = []
        all_unit_dones_list = []
        batch_indices = []  # Track which batch item each unit belongs to
        
        # Fix action_mask shape
        action_mask = batch['action_mask']
        if action_mask is not None:
            if len(action_mask.shape) == 5:
                if action_mask.size(1) == 1:
                    action_mask = action_mask.squeeze(1)
                elif action_mask.shape[0] == action_mask.shape[1]:
                    action_mask = action_mask[:, 0, :, :, :]
        action_mask_np = action_mask.cpu().numpy() if action_mask is not None else None
        
        # Handle action shape
        actions = batch['action']
        if actions.dim() == 4:
            if actions.size(1) == 1:
                actions = actions.squeeze(1)
        actions_np = actions.cpu().numpy()  # (batch, H, W)
        
        # Extract local observations for each batch item
        for b in range(batch_size):
            grid_b = obs_np[b]  # (3, H, W)
            local_obs_b, unit_positions_b = extract_local_observations(grid_b, agent_id)
            num_units_b = len(unit_positions_b)
            
            if num_units_b == 0:
                continue
            
            # Extract actions and action masks for units
            unit_positions_array = np.array(unit_positions_b)
            y_coords = unit_positions_array[:, 0]
            x_coords = unit_positions_array[:, 1]
            
            unit_actions_b = actions_np[b, y_coords, x_coords]  # (num_units,)
            
            unit_action_masks_b = None
            if action_mask_np is not None:
                unit_action_masks_b = action_mask_np[b, y_coords, x_coords]  # (num_units, 5)
            
            all_unit_obs_list.append(local_obs_b)
            all_unit_actions_list.append(unit_actions_b)
            all_unit_action_masks_list.append(unit_action_masks_b)
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
        all_unit_action_masks_tensor = None
        if all_unit_action_masks is not None:
            all_unit_action_masks_tensor = torch.BoolTensor(all_unit_action_masks).unsqueeze(0).to(self.device)  # (1, total_units, 5)
        
        # Current Q-values
        q_values = q_net(all_unit_obs_tensor, all_unit_action_masks_tensor)  # (1, total_units, 5)
        q_values = q_values.squeeze(0)  # (total_units, 5)
        
        # Gather Q-values for taken actions
        q_selected = torch.gather(
            q_values,
            dim=1,
            index=all_unit_actions_tensor.unsqueeze(-1)
        ).squeeze(-1)  # (total_units,)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_obs = batch['next_obs']
            next_obs_shape = next_obs.shape
            
            if len(next_obs_shape) == 5:
                if next_obs_shape[1] == 1:
                    next_obs = next_obs.squeeze(1)
                elif next_obs_shape[0] == next_obs_shape[1]:
                    next_obs = next_obs[:, 0, :, :, :]
            elif len(next_obs_shape) != 4:
                print(f"Warning: Unexpected next_obs shape: {next_obs_shape}")
                return None
            
            next_obs_np = next_obs.cpu().numpy()  # (batch, 3, H, W)
            
            # Extract local observations for next states
            all_next_unit_obs_list = []
            all_next_unit_action_masks_list = []
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
                next_local_obs_b, next_unit_positions_b = extract_local_observations(next_grid_b, agent_id)
                next_num_units_b = len(next_unit_positions_b)
                
                if next_num_units_b == 0:
                    continue
                
                next_unit_action_masks_b = None
                if next_action_mask_np is not None:
                    next_unit_positions_array = np.array(next_unit_positions_b)
                    next_y_coords = next_unit_positions_array[:, 0]
                    next_x_coords = next_unit_positions_array[:, 1]
                    next_unit_action_masks_b = next_action_mask_np[b, next_y_coords, next_x_coords]  # (next_num_units, 5)
                
                all_next_unit_obs_list.append(next_local_obs_b)
                all_next_unit_action_masks_list.append(next_unit_action_masks_b)
                next_batch_indices.extend([b] * next_num_units_b)
            
            if len(all_next_unit_obs_list) == 0:
                # No units in next state, set target to reward only
                next_q_max_flat = torch.zeros_like(q_selected)
            else:
                all_next_unit_obs = np.concatenate(all_next_unit_obs_list, axis=0)  # (total_next_units, 7, 7, 6)
                all_next_unit_action_masks = None
                if all_next_unit_action_masks_list[0] is not None:
                    all_next_unit_action_masks = np.concatenate([m for m in all_next_unit_action_masks_list if m is not None], axis=0)
                
                all_next_unit_obs_tensor = torch.FloatTensor(all_next_unit_obs).unsqueeze(0).to(self.device)  # (1, total_next_units, 7, 7, 6)
                all_next_unit_action_masks_tensor = None
                if all_next_unit_action_masks is not None:
                    all_next_unit_action_masks_tensor = torch.BoolTensor(all_next_unit_action_masks).unsqueeze(0).to(self.device)  # (1, total_next_units, 5)
                
                next_q_values = target_q_net(all_next_unit_obs_tensor, all_next_unit_action_masks_tensor)  # (1, total_next_units, 5)
                next_q_values = next_q_values.squeeze(0)  # (total_next_units, 5)
                
                # Handle -inf from masking
                next_q_values_clamped = next_q_values.clone()
                next_q_values_clamped[next_q_values_clamped == float('-inf')] = -1e6
                next_q_max = next_q_values_clamped.max(dim=-1)[0]  # (total_next_units,)
                
                # Map back to original units (handle case where number of units changed)
                # For simplicity, we'll use the max Q-value per batch item
                next_q_max_by_batch = {}
                for idx, b_idx in enumerate(next_batch_indices):
                    if b_idx not in next_q_max_by_batch:
                        next_q_max_by_batch[b_idx] = []
                    next_q_max_by_batch[b_idx].append(next_q_max[idx].item())
                
                # Create next_q_max_flat matching original units
                next_q_max_flat = torch.zeros_like(q_selected)
                for idx, b_idx in enumerate(batch_indices):
                    if b_idx in next_q_max_by_batch and len(next_q_max_by_batch[b_idx]) > 0:
                        next_q_max_flat[idx] = max(next_q_max_by_batch[b_idx])
            
            # Target Q-values
            rewards = batch['reward']
            if rewards.dim() == 2 and rewards.size(1) == 1:
                rewards = rewards.squeeze(1)  # (batch,)
            elif rewards.dim() == 2 and rewards.size(1) > 1:
                rewards = rewards[:, 0]
            elif rewards.dim() == 1:
                pass
            else:
                print(f"Warning: Unexpected reward shape: {rewards.shape}")
                return None
            
            done = batch['done']
            if done.dim() == 2:
                done = done.squeeze(1) if done.size(1) == 1 else done[:, 0]
            elif done.dim() > 2:
                print(f"Warning: Unexpected done shape: {done.shape}")
                return None
            
            # Expand rewards and done to match units
            rewards_expanded = torch.zeros_like(q_selected)
            done_expanded = torch.zeros_like(q_selected, dtype=torch.float)
            for idx, b_idx in enumerate(batch_indices):
                rewards_expanded[idx] = rewards[b_idx]
                done_expanded[idx] = done[b_idx].float()
            
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
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
        else:
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Skipping IQL update due to invalid loss: {loss.item()}")
            elif loss.item() >= 1e10:
                print(f"Warning: Skipping IQL update due to very large loss: {loss.item()}")
        
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
    ) -> List[Optional[Dict[str, float]]]:
        """Add transitions to buffers and update all agents."""
        grid = obs['grid']
        
        # Store transition for each agent separately
        updates = []
        for agent_id in range(self.env.num_agents):
            agent_ownership = grid[0] == (agent_id + 1)
            
            if not np.any(agent_ownership):
                continue
            
            # Extract agent-specific action (only for owned cells)
            agent_action = action[agent_id:agent_id+1].copy()
            
            # Add to agent's replay buffer
            self.replay_buffers[agent_id].push(
                obs=obs['grid'],
                action=agent_action,
                reward=reward[agent_id:agent_id+1],
                next_obs=next_obs['grid'],
                done=done,
                action_mask=obs['action_mask'][agent_id:agent_id+1],
                next_action_mask=next_obs['action_mask'][agent_id:agent_id+1]
            )
            
            # Update agent
            update_stats = self.update(agent_id)
            updates.append(update_stats)
        
        # Update target networks periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            if self.shared_network:
                self.target_q_networks.load_state_dict(self.q_networks.state_dict())
            else:
                for i in range(self.env.num_agents):
                    self.target_q_networks[i].load_state_dict(
                        self.q_networks[i].state_dict()
                    )
        
        # Decay epsilon (slower decay to address non-stationarity)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return updates
    
    def save(self, path: str):
        """Save model weights."""
        save_dict = {
            'q_networks': self.q_networks.state_dict(),
            'target_q_networks': self.target_q_networks.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'shared_network': self.shared_network
        }
        
        if self.shared_network:
            save_dict['optimizer'] = self.optimizers.state_dict()
        else:
            save_dict['optimizers'] = [
                opt.state_dict() for opt in self.optimizers
            ]
        
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_networks.load_state_dict(checkpoint['q_networks'])
        self.target_q_networks.load_state_dict(checkpoint['target_q_networks'])
        
        if self.shared_network:
            self.optimizers.load_state_dict(checkpoint['optimizer'])
        else:
            for i, opt in enumerate(self.optimizers):
                opt.load_state_dict(checkpoint['optimizers'][i])
        
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']

