"""
Unit tests for the RuleBasedBot focusing on sanity checks.
Tests that the bot never produces invalid moves.
"""
import pytest
import numpy as np
from src.bots.rule_based import RuleBasedBot
from src.env.env import Halite
from src.env.constants import _NUM_ACTIONS


class TestRuleBasedBotSanity:
    """Test suite for RuleBasedBot sanity checks."""

    def test_no_actions_on_unowned_cells(self):
        """Test that bot never produces actions on cells it doesn't own."""
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        # Test with multiple random seeds
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            grid = obs["grid"]
            
            # Get bot's actions
            actions = bot(grid, seed=seed)
            
            # Get ownership map for this agent (1-based)
            agent_id_1b = bot.agent_id + 1
            ownership = grid[0]
            owned_cells = (ownership == agent_id_1b)
            
            # All actions on unowned cells must be 0 (WAIT)
            unowned_actions = actions[~owned_cells]
            assert np.all(unowned_actions == 0), \
                f"Found non-zero actions on unowned cells: {unowned_actions[unowned_actions != 0]}"

    def test_actions_in_valid_range(self):
        """Test that all actions are in the valid range [0, _NUM_ACTIONS-1]."""
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        # Test with multiple random seeds
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            grid = obs["grid"]
            
            # Get bot's actions
            actions = bot(grid, seed=seed)
            
            # All actions must be in valid range
            assert np.all(actions >= 0), f"Found negative actions: {actions[actions < 0]}"
            assert np.all(actions < _NUM_ACTIONS), \
                f"Found actions >= {_NUM_ACTIONS}: {actions[actions >= _NUM_ACTIONS]}"

    def test_action_shape_matches_observation(self):
        """Test that action shape matches the observation grid dimensions."""
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        
        # Test with different grid sizes
        grid_sizes = [(3, 3), (5, 5), (7, 7), (10, 10)]
        
        for grid_size in grid_sizes:
            env = Halite(num_agents=2, grid_size=grid_size)
            obs, _ = env.reset(seed=42)
            grid = obs["grid"]
            
            actions = bot(grid, seed=42)
            
            # Actions should have shape matching the actual observation grid dimensions
            ownership = grid[0]
            expected_height, expected_width = ownership.shape
            assert actions.shape == (expected_height, expected_width), \
                f"Expected shape ({expected_height}, {expected_width}), got {actions.shape}"

    def test_bot_handles_empty_ownership(self):
        """Test that bot handles cases where agent has no owned cells gracefully."""
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        obs, _ = env.reset(seed=42)
        grid = obs["grid"]
        
        # Get actual grid dimensions
        height, width = grid[0].shape
        
        # Manually set all cells to be owned by agent 1 (not agent 0)
        grid[0, :, :] = 2  # All cells owned by agent 2
        
        # Bot should still return valid actions (all WAIT on unowned cells)
        actions = bot(grid, seed=42)
        
        # All actions should be 0 (WAIT) since agent 0 owns nothing
        assert np.all(actions == 0), "Bot should return all WAIT actions when it owns no cells"
        assert actions.shape == (height, width), \
            f"Actions should have correct shape ({height}, {width}), got {actions.shape}"

    def test_bot_consistency_with_same_seed(self):
        """Test that bot produces consistent actions with the same seed."""
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        obs, _ = env.reset(seed=42)
        grid = obs["grid"]
        
        # Get actions twice with the same seed
        actions1 = bot(grid, seed=123)
        actions2 = bot(grid, seed=123)
        
        # Actions should be identical with the same seed
        np.testing.assert_array_equal(actions1, actions2)

    def test_bot_with_multiple_agents(self):
        """Test that bot works correctly in multi-agent scenarios."""
        num_agents = 3
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        env = Halite(num_agents=num_agents, grid_size=(5, 5))
        
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            grid = obs["grid"]
            
            actions = bot(grid, seed=seed)
            
            # Check validity
            assert np.all(actions >= 0) and np.all(actions < _NUM_ACTIONS)
            
            # Check that actions on unowned cells are 0
            agent_id_1b = bot.agent_id + 1
            ownership = grid[0]
            owned_cells = (ownership == agent_id_1b)
            unowned_actions = actions[~owned_cells]
            assert np.all(unowned_actions == 0)

    def test_bot_through_full_episode(self):
        """Test that bot produces valid actions throughout a full episode."""
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        obs, _ = env.reset(seed=42)
        terminated = False
        step_count = 0
        max_steps = 50
        
        while not terminated and step_count < max_steps:
            grid = obs["grid"]
            actions_bot = bot(grid, seed=42 + step_count)
            
            # Get actual grid dimensions
            height, width = grid[0].shape
            
            # Check action validity
            assert np.all(actions_bot >= 0) and np.all(actions_bot < _NUM_ACTIONS)
            
            # Check that actions on unowned cells are 0
            agent_id_1b = bot.agent_id + 1
            ownership = grid[0]
            owned_cells = (ownership == agent_id_1b)
            unowned_actions = actions_bot[~owned_cells]
            assert np.all(unowned_actions == 0), \
                f"Step {step_count}: Found invalid actions on unowned cells"
            
            # Create full action array for environment (all agents)
            full_actions = np.zeros((env.num_agents, height, width), dtype=np.int32)
            full_actions[0] = actions_bot  # Use bot's actions for agent 0
            # Other agents do nothing for this test
            
            obs, reward, terminated, truncated, info = env.step(full_actions)
            step_count += 1

    def test_bot_actions_match_action_mask(self):
        """Test that bot's actions respect the action mask (only act on owned cells)."""
        bot = RuleBasedBot(agent_id=0, name="TestBot")
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            grid = obs["grid"]
            action_mask = obs["action_mask"]
            
            actions = bot(grid, seed=seed)
            
            # Get action mask for this agent
            agent_mask = action_mask[bot.agent_id]  # (H, W, num_actions)
            
            # For each cell, if bot produced a non-zero action, check it's allowed
            for y in range(actions.shape[0]):
                for x in range(actions.shape[1]):
                    action = actions[y, x]
                    if action != 0:
                        # This cell must be owned by the agent
                        agent_id_1b = bot.agent_id + 1
                        assert grid[0, y, x] == agent_id_1b, \
                            f"Bot produced action {action} on unowned cell ({y}, {x})"
                        # Action must be in valid range
                        assert 0 <= action < _NUM_ACTIONS, \
                            f"Bot produced invalid action {action} at ({y}, {x})"

