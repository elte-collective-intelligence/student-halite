"""Neural network architectures for multi-agent RL algorithms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CNNFeatureExtractor(nn.Module):
    """CNN feature extractor for grid-based observations."""
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 256
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Adaptive pooling to handle different grid sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            features: (batch, output_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x


class QNetwork(nn.Module):
    """Q-network for Q-learning algorithms.
    
    Uses fully convolutional architecture to handle variable grid sizes.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        grid_size: Tuple[int, int] = (5, 5),
        num_actions: int = 5,
        hidden_dim: int = 128,
        feature_dim: int = 256
    ):
        super().__init__()
        self.grid_size = grid_size  # Used for reference, but network is adaptive
        self.num_actions = num_actions
        
        # Fully convolutional architecture for variable grid sizes
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Output Q-values for each cell and action (spatial)
        self.q_head = nn.Conv2d(256, num_actions, kernel_size=1)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            obs: (batch, channels, height, width)
            action_mask: (batch, height, width, num_actions) - optional mask for valid actions
        Returns:
            q_values: (batch, height, width, num_actions)
        """
        batch_size = obs.size(0)
        H, W = obs.size(2), obs.size(3)  # Get actual dimensions from input
        
        # Fully convolutional forward pass
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x: (batch, 256, H, W)
        
        # Output Q-values per spatial location
        q_spatial = self.q_head(x)  # (batch, num_actions, H, W)
        q_values = q_spatial.permute(0, 2, 3, 1)  # (batch, H, W, num_actions)
        
        if action_mask is not None:
            # Mask invalid actions with large negative values
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        return q_values


class CentralizedQNetwork(nn.Module):
    """Centralized Q-network for joint state-action value estimation.
    
    Uses fully convolutional architecture to handle variable grid sizes.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        grid_size: Tuple[int, int] = (5, 5),
        num_agents: int = 2,
        num_actions: int = 5,
        hidden_dim: int = 128,
        feature_dim: int = 512
    ):
        super().__init__()
        self.grid_size = grid_size  # Used for reference, but network is adaptive
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        # Fully convolutional architecture for variable grid sizes
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Output Q-values for each agent, cell, and action (spatial)
        self.q_head = nn.Conv2d(
            256,
            num_agents * num_actions,
            kernel_size=1
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights with small values."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            obs: (batch, channels, height, width)
            action_mask: (batch, num_agents, height, width, num_actions) - optional
        Returns:
            q_values: (batch, num_agents, height, width, num_actions)
        """
        batch_size = obs.size(0)
        H, W = obs.size(2), obs.size(3)  # Get actual dimensions from input
        
        # Fully convolutional forward pass
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x: (batch, 256, H, W)
        
        # Output Q-values per spatial location
        q_spatial = self.q_head(x)  # (batch, num_agents*num_actions, H, W)
        q_values = q_spatial.view(
            batch_size, self.num_agents, self.num_actions, H, W
        )
        q_values = q_values.permute(0, 1, 3, 4, 2)  # (batch, num_agents, H, W, num_actions)
        
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        return q_values


class ActorNetwork(nn.Module):
    """Actor network for policy gradient methods.
    
    Uses fully convolutional architecture to handle variable grid sizes.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        grid_size: Tuple[int, int] = (5, 5),  # Reference size, but network is adaptive
        num_actions: int = 5,
        hidden_dim: int = 128,
        feature_dim: int = 256
    ):
        super().__init__()
        self.grid_size = grid_size  # Kept for compatibility, but not used in forward
        self.num_actions = num_actions
        
        # Fully convolutional feature extraction and policy head
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim, feature_dim, kernel_size=3, padding=1)
        
        # Output action logits per spatial location
        self.policy_head = nn.Conv2d(feature_dim, num_actions, kernel_size=1)
        
    def forward(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Args:
            obs: (batch, channels, height, width)
            action_mask: (batch, height, width, num_actions) - optional
        Returns:
            action_dist: Categorical distribution over actions
            logits: (batch, height, width, num_actions)
        """
        batch_size = obs.size(0)
        H, W = obs.size(2), obs.size(3)  # Get actual dimensions from input
        
        # Fully convolutional forward pass
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # x: (batch, feature_dim, H, W)
        
        # Output action logits per spatial location
        logits_spatial = self.policy_head(x)  # (batch, num_actions, H, W)
        logits = logits_spatial.permute(0, 2, 3, 1)  # (batch, H, W, num_actions)
        
        if action_mask is not None:
            # Ensure action_mask has correct dimensions: (batch, H, W, num_actions)
            if action_mask.dim() == 5:
                # (batch, 1, H, W, num_actions) -> squeeze agent dimension
                if action_mask.size(1) == 1:
                    action_mask = action_mask.squeeze(1)
                else:
                    # Take first agent's mask
                    action_mask = action_mask[:, 0, :, :, :]
            elif action_mask.dim() != 4:
                raise ValueError(f"Unexpected action_mask dimensions: {action_mask.shape}")
            
            # Check for cells where all actions are masked
            all_masked = ~action_mask.any(dim=-1)  # (batch, H, W)
            
            # For cells with all actions masked, allow at least the first action
            action_mask_safe = action_mask.clone()
            # Set first action to True where all actions were masked
            action_mask_safe[:, :, :, 0] = action_mask_safe[:, :, :, 0] | all_masked
            
            # Mask invalid actions (use safe mask to prevent all -inf)
            logits = logits.masked_fill(~action_mask_safe, float('-inf'))
        
        # Replace -inf with large negative value to prevent NaN in Categorical
        logits = logits.masked_fill(torch.isinf(logits) & (logits < 0), -1e6)
        
        # Replace any NaN values (shouldn't happen, but safety check)
        logits = torch.where(torch.isnan(logits), torch.tensor(-1e6, device=logits.device), logits)
        
        # Create independent categorical distributions for each cell
        # Reshape to (batch, H*W, num_actions) for Categorical distribution
        logits_reshaped = logits.view(
            batch_size, H * W, self.num_actions
        )
        
        # Final safety check: ensure no NaN or Inf before creating distribution
        logits_reshaped = torch.clamp(logits_reshaped, min=-1e6, max=1e6)
        logits_reshaped = torch.where(
            torch.isnan(logits_reshaped) | torch.isinf(logits_reshaped),
            torch.zeros_like(logits_reshaped),
            logits_reshaped
        )
        
        dist = torch.distributions.Categorical(logits=logits_reshaped)
        
        return dist, logits


class LocalActorNetwork(nn.Module):
    """Actor network for processing local 7x7 patches.
    
    Processes individual unit observations (7x7 patches with 6 channels)
    and outputs action logits for each unit.
    """
    
    def __init__(
        self,
        input_channels: int = 6,  # 6 channels: is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask
        patch_size: int = 7,
        num_actions: int = 5,
        hidden_dim: int = 128,
        feature_dim: int = 256
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_actions = num_actions
        
        # Process 7x7 patches with 6 channels
        # Use smaller kernels since patch is small
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1)
        
        # Global average pooling to get a single feature vector per patch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to process pooled features
        self.fc1 = nn.Linear(hidden_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_actions)
        
    def forward(
        self,
        local_patches: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Args:
            local_patches: (batch, num_units, patch_size, patch_size, input_channels)
                          or (batch * num_units, input_channels, patch_size, patch_size)
            action_mask: (batch, num_units, num_actions) - optional mask for valid actions
        Returns:
            action_dist: Categorical distribution over actions
            logits: (batch, num_units, num_actions) or (batch * num_units, num_actions)
        """
        # Handle both input formats
        if local_patches.dim() == 5:
            # (batch, num_units, patch_size, patch_size, channels)
            batch_size, num_units, H, W, C = local_patches.shape
            # Reshape to (batch * num_units, channels, H, W)
            local_patches = local_patches.view(batch_size * num_units, C, H, W)
            reshape_output = True
        else:
            # (batch * num_units, channels, H, W)
            reshape_output = False
        
        # Convolutional feature extraction
        x = F.relu(self.conv1(local_patches))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling: (batch * num_units, hidden_dim, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch * num_units, hidden_dim)
        
        # MLP to get action logits
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)  # (batch * num_units, num_actions)
        
        if reshape_output:
            # Reshape back to (batch, num_units, num_actions)
            logits = logits.view(batch_size, num_units, self.num_actions)
        
        if action_mask is not None:
            # Ensure action_mask matches logits shape
            if action_mask.dim() == 3 and logits.dim() == 3:
                # action_mask: (batch, num_units, num_actions)
                # logits: (batch, num_units, num_actions)
                logits = logits.masked_fill(~action_mask, float('-inf'))
            elif action_mask.dim() == 2 and logits.dim() == 3:
                # action_mask: (num_units, num_actions) - squeeze batch dimension
                # logits: (batch, num_units, num_actions)
                if action_mask.size(0) == logits.size(1):
                    action_mask = action_mask.unsqueeze(0)  # (1, num_units, num_actions)
                    logits = logits.masked_fill(~action_mask, float('-inf'))
            elif action_mask.dim() == 2 and logits.dim() == 2:
                # Both flattened: (batch * num_units, num_actions)
                logits = logits.masked_fill(~action_mask, float('-inf'))
        
        # Replace -inf with large negative value to prevent NaN
        logits = logits.masked_fill(torch.isinf(logits) & (logits < 0), -1e6)
        logits = torch.where(torch.isnan(logits), torch.tensor(-1e6, device=logits.device), logits)
        
        # Clamp to prevent extreme values
        logits = torch.clamp(logits, min=-1e6, max=1e6)
        
        # Create categorical distribution
        if logits.dim() == 3:
            # (batch, num_units, num_actions) -> flatten to (batch * num_units, num_actions)
            logits_flat = logits.view(-1, self.num_actions)
            dist = torch.distributions.Categorical(logits=logits_flat)
        else:
            dist = torch.distributions.Categorical(logits=logits)
        
        return dist, logits


class LocalQNetwork(nn.Module):
    """Q-network for processing local 7x7 patches.
    
    Processes individual unit observations (7x7 patches with 6 channels)
    and outputs Q-values for each unit.
    """
    
    def __init__(
        self,
        input_channels: int = 6,  # 6 channels: is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask
        patch_size: int = 7,
        num_actions: int = 5,
        hidden_dim: int = 128,
        feature_dim: int = 256
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_actions = num_actions
        
        # Process 7x7 patches with 6 channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1)
        
        # Global average pooling to get a single feature vector per patch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to process pooled features and output Q-values
        self.fc1 = nn.Linear(hidden_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_actions)
        
    def forward(
        self,
        local_patches: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            local_patches: (batch, num_units, patch_size, patch_size, input_channels)
                          or (batch * num_units, input_channels, patch_size, patch_size)
            action_mask: (batch, num_units, num_actions) - optional mask for valid actions
        Returns:
            q_values: (batch, num_units, num_actions) or (batch * num_units, num_actions)
        """
        # Handle both input formats
        if local_patches.dim() == 5:
            # (batch, num_units, patch_size, patch_size, channels)
            batch_size, num_units, H, W, C = local_patches.shape
            # Reshape to (batch * num_units, channels, H, W)
            local_patches = local_patches.view(batch_size * num_units, C, H, W)
            reshape_output = True
        else:
            # (batch * num_units, channels, H, W)
            reshape_output = False
        
        # Convolutional feature extraction
        x = F.relu(self.conv1(local_patches))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling: (batch * num_units, hidden_dim, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch * num_units, hidden_dim)
        
        # MLP to get Q-values
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)  # (batch * num_units, num_actions)
        
        if reshape_output:
            # Reshape back to (batch, num_units, num_actions)
            q_values = q_values.view(batch_size, num_units, self.num_actions)
        
        if action_mask is not None:
            # Ensure action_mask matches q_values shape
            if action_mask.dim() == 3 and q_values.dim() == 3:
                # action_mask: (batch, num_units, num_actions)
                # q_values: (batch, num_units, num_actions)
                q_values = q_values.masked_fill(~action_mask, float('-inf'))
            elif action_mask.dim() == 2 and q_values.dim() == 3:
                # action_mask: (num_units, num_actions) - squeeze batch dimension
                if action_mask.size(0) == q_values.size(1):
                    action_mask = action_mask.unsqueeze(0)  # (1, num_units, num_actions)
                    q_values = q_values.masked_fill(~action_mask, float('-inf'))
            elif action_mask.dim() == 2 and q_values.dim() == 2:
                # Both flattened: (batch * num_units, num_actions)
                q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        # Replace -inf with large negative value to prevent NaN
        q_values = q_values.masked_fill(torch.isinf(q_values) & (q_values < 0), -1e6)
        q_values = torch.where(torch.isnan(q_values), torch.tensor(-1e6, device=q_values.device), q_values)
        
        # Clamp to prevent extreme values
        q_values = torch.clamp(q_values, min=-1e6, max=1e6)
        
        return q_values


class LocalCentralizedQNetwork(nn.Module):
    """Centralized Q-network for processing local 7x7 patches.
    
    Processes individual unit observations (7x7 patches with 6 channels)
    and outputs Q-values for each agent and unit.
    """
    
    def __init__(
        self,
        input_channels: int = 6,  # 6 channels: is_mine, is_enemy, is_neutral, normalized_strength, normalized_production, unit_mask
        patch_size: int = 7,
        num_agents: int = 2,
        num_actions: int = 5,
        hidden_dim: int = 128,
        feature_dim: int = 256
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        # Process 7x7 patches with 6 channels
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, hidden_dim, kernel_size=3, padding=1)
        
        # Global average pooling to get a single feature vector per patch
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP to process pooled features and output Q-values for all agents
        self.fc1 = nn.Linear(hidden_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, num_agents * num_actions)
        
    def forward(
        self,
        local_patches: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            local_patches: (batch, num_units, patch_size, patch_size, input_channels)
                          or (batch * num_units, input_channels, patch_size, patch_size)
            action_mask: (batch, num_agents, num_units, num_actions) - optional mask for valid actions
        Returns:
            q_values: (batch, num_agents, num_units, num_actions)
        """
        # Handle both input formats
        if local_patches.dim() == 5:
            # (batch, num_units, patch_size, patch_size, channels)
            batch_size, num_units, H, W, C = local_patches.shape
            # Reshape to (batch * num_units, channels, H, W)
            local_patches = local_patches.view(batch_size * num_units, C, H, W)
            reshape_output = True
        else:
            # (batch * num_units, channels, H, W)
            reshape_output = False
        
        # Convolutional feature extraction
        x = F.relu(self.conv1(local_patches))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Global average pooling: (batch * num_units, hidden_dim, 1, 1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # (batch * num_units, hidden_dim)
        
        # MLP to get Q-values for all agents
        x = F.relu(self.fc1(x))
        q_values_flat = self.fc2(x)  # (batch * num_units, num_agents * num_actions)
        
        if reshape_output:
            # Reshape to (batch, num_units, num_agents, num_actions)
            q_values = q_values_flat.view(batch_size, num_units, self.num_agents, self.num_actions)
            # Permute to (batch, num_agents, num_units, num_actions)
            q_values = q_values.permute(0, 2, 1, 3)
        else:
            # Reshape to (batch * num_units, num_agents, num_actions)
            q_values = q_values_flat.view(-1, self.num_agents, self.num_actions)
        
        if action_mask is not None:
            # Ensure action_mask matches q_values shape
            if action_mask.dim() == 4:
                # action_mask: (batch, num_agents, num_units, num_actions)
                # q_values: (batch, num_agents, num_units, num_actions)
                q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        # Replace -inf with large negative value to prevent NaN
        q_values = q_values.masked_fill(torch.isinf(q_values) & (q_values < 0), -1e6)
        q_values = torch.where(torch.isnan(q_values), torch.tensor(-1e6, device=q_values.device), q_values)
        
        # Clamp to prevent extreme values
        q_values = torch.clamp(q_values, min=-1e6, max=1e6)
        
        return q_values


class CriticNetwork(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(
        self,
        input_channels: int = 3,
        grid_size: Tuple[int, int] = (5, 5),
        hidden_dim: int = 128,
        feature_dim: int = 256,
        centralized: bool = False,
        num_agents: int = 1
    ):
        super().__init__()
        self.centralized = centralized
        
        if centralized:
            # For MAPPO: centralized critic sees global state
            self.feature_extractor = CNNFeatureExtractor(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                output_dim=feature_dim
            )
            # Output value for each agent
            self.value_head = nn.Linear(feature_dim, num_agents)
        else:
            # For IPPO: independent critic per agent
            self.feature_extractor = CNNFeatureExtractor(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                output_dim=feature_dim
            )
            # Output single scalar value
            self.value_head = nn.Linear(feature_dim, 1)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, channels, height, width)
        Returns:
            values: (batch, num_agents) if centralized else (batch, 1)
        """
        features = self.feature_extractor(obs)
        values = self.value_head(features)
        return values

