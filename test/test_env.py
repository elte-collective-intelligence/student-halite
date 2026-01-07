"""
Unit tests for the Halite environment focusing on determinism.
Tests that reset() and step() produce deterministic results with fixed seeds.
"""
import pytest
import numpy as np
from src.env.env import Halite


class TestEnvironmentDeterminism:
    """Test suite for environment determinism with fixed seeds."""

    def test_reset_determinism(self):
        """Test that reset() produces identical states with the same seed."""
        env1 = Halite(num_agents=2, grid_size=(5, 5))
        env2 = Halite(num_agents=2, grid_size=(5, 5))
        
        seed = 42
        obs1, info1 = env1.reset(seed=seed)
        obs2, info2 = env2.reset(seed=seed)
        
        # Check that observations are identical
        np.testing.assert_array_equal(obs1["grid"], obs2["grid"])
        np.testing.assert_array_equal(obs1["action_mask"], obs2["action_mask"])
        assert obs1["step_count"] == obs2["step_count"]
        
        # Check that internal states are identical
        np.testing.assert_array_equal(env1.state.grid, env2.state.grid)
        np.testing.assert_array_equal(env1.state.alive, env2.state.alive)
        assert env1.state.step_count == env2.state.step_count

    def test_reset_determinism_multiple_calls(self):
        """Test that multiple reset() calls with the same seed produce identical results."""
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        seed = 123
        obs1, _ = env.reset(seed=seed)
        state1 = env.state
        
        obs2, _ = env.reset(seed=seed)
        state2 = env.state
        
        # Check that observations are identical
        np.testing.assert_array_equal(obs1["grid"], obs2["grid"])
        np.testing.assert_array_equal(obs1["action_mask"], obs2["action_mask"])
        assert obs1["step_count"] == obs2["step_count"]
        
        # Check that internal states are identical
        np.testing.assert_array_equal(state1.grid, state2.grid)
        np.testing.assert_array_equal(state1.alive, state2.alive)
        assert state1.step_count == state2.step_count

    def test_step_determinism(self):
        """Test that step() produces identical results with the same seed and actions."""
        env1 = Halite(num_agents=2, grid_size=(5, 5))
        env2 = Halite(num_agents=2, grid_size=(5, 5))
        
        seed = 42
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Verify initial states are identical
        np.testing.assert_array_equal(env1.state.grid, env2.state.grid)
        
        # Get actual grid dimensions from observation
        ownership = obs1["grid"][0]
        height, width = ownership.shape
        
        # Create deterministic actions only on owned cells
        actions = np.zeros((2, height, width), dtype=np.int32)
        
        # Find owned cells for each agent and set actions only there
        for agent_id in range(2):
            agent_id_1b = agent_id + 1
            owned_cells = np.where(ownership == agent_id_1b)
            if len(owned_cells[0]) > 0:
                # Set action on first owned cell
                y, x = owned_cells[0][0], owned_cells[1][0]
                actions[agent_id, y, x] = 1  # Move UP
        
        # Take step in both environments
        obs1_next, reward1, terminated1, truncated1, info1 = env1.step(actions)
        obs2_next, reward2, terminated2, truncated2, info2 = env2.step(actions)
        
        # Check that observations are identical
        np.testing.assert_array_equal(obs1_next["grid"], obs2_next["grid"])
        np.testing.assert_array_equal(obs1_next["action_mask"], obs2_next["action_mask"])
        assert obs1_next["step_count"] == obs2_next["step_count"]
        
        # Check that rewards are identical
        np.testing.assert_array_equal(reward1, reward2)
        
        # Check that termination flags are identical
        assert terminated1 == terminated2
        assert truncated1 == truncated2
        
        # Check that internal states are identical
        np.testing.assert_array_equal(env1.state.grid, env2.state.grid)
        np.testing.assert_array_equal(env1.state.alive, env2.state.alive)
        assert env1.state.step_count == env2.state.step_count

    def test_multiple_steps_determinism(self):
        """Test that multiple steps produce identical results with the same seed."""
        env1 = Halite(num_agents=2, grid_size=(5, 5))
        env2 = Halite(num_agents=2, grid_size=(5, 5))
        
        seed = 42
        obs1 = env1.reset(seed=seed)[0]
        obs2 = env2.reset(seed=seed)[0]
        
        # Get actual grid dimensions
        ownership = obs1["grid"][0]
        height, width = ownership.shape
        
        # Take multiple steps with deterministic actions
        num_steps = 5
        for step in range(num_steps):
            # Create deterministic actions only on owned cells
            actions = np.zeros((2, height, width), dtype=np.int32)
            
            # Set actions only on owned cells
            for agent_id in range(2):
                agent_id_1b = agent_id + 1
                owned_cells = np.where(ownership == agent_id_1b)
                if len(owned_cells[0]) > 0:
                    # Cycle through owned cells and directions
                    idx = step % len(owned_cells[0])
                    y, x = owned_cells[0][idx], owned_cells[1][idx]
                    actions[agent_id, y, x] = (step % 4) + 1  # Cycle through directions
            
            obs1, reward1, terminated1, _, _ = env1.step(actions)
            obs2, reward2, terminated2, _, _ = env2.step(actions)
            
            # Update ownership for next iteration
            ownership = obs1["grid"][0]
            
            # Check that states remain identical after each step
            np.testing.assert_array_equal(env1.state.grid, env2.state.grid)
            np.testing.assert_array_equal(env1.state.alive, env2.state.alive)
            assert env1.state.step_count == env2.state.step_count
            np.testing.assert_array_equal(reward1, reward2)
            assert terminated1 == terminated2
            
            if terminated1:
                break

    def test_different_seeds_produce_different_states(self):
        """Test that different seeds produce different initial states."""
        env = Halite(num_agents=2, grid_size=(5, 5))
        
        obs1, _ = env.reset(seed=42)
        state1 = env.state
        
        obs2, _ = env.reset(seed=123)
        state2 = env.state
        
        # States should be different (very unlikely to be identical with different seeds)
        assert not np.array_equal(state1.grid, state2.grid)

    def test_determinism_with_different_grid_sizes(self):
        """Test determinism works with different grid sizes."""
        grid_sizes = [(3, 3), (5, 5), (7, 7)]
        
        for grid_size in grid_sizes:
            env1 = Halite(num_agents=2, grid_size=grid_size)
            env2 = Halite(num_agents=2, grid_size=grid_size)
            
            seed = 42
            obs1, _ = env1.reset(seed=seed)
            obs2, _ = env2.reset(seed=seed)
            
            np.testing.assert_array_equal(obs1["grid"], obs2["grid"])
            np.testing.assert_array_equal(env1.state.grid, env2.state.grid)

    def test_determinism_with_different_num_agents(self):
        """Test determinism works with different numbers of agents."""
        num_agents_list = [2, 3, 4]
        
        for num_agents in num_agents_list:
            env1 = Halite(num_agents=num_agents, grid_size=(5, 5))
            env2 = Halite(num_agents=num_agents, grid_size=(5, 5))
            
            seed = 42
            obs1, _ = env1.reset(seed=seed)
            obs2, _ = env2.reset(seed=seed)
            
            np.testing.assert_array_equal(obs1["grid"], obs2["grid"])
            np.testing.assert_array_equal(env1.state.grid, env2.state.grid)
            
            # Get actual grid dimensions
            ownership = obs1["grid"][0]
            height, width = ownership.shape
            
            # Test step determinism - create actions only on owned cells
            actions = np.zeros((num_agents, height, width), dtype=np.int32)
            
            # Set actions only on owned cells
            for agent_id in range(num_agents):
                agent_id_1b = agent_id + 1
                owned_cells = np.where(ownership == agent_id_1b)
                if len(owned_cells[0]) > 0:
                    # Set action on first owned cell
                    y, x = owned_cells[0][0], owned_cells[1][0]
                    actions[agent_id, y, x] = 1  # Move UP
            
            obs1_next, reward1, term1, _, _ = env1.step(actions)
            obs2_next, reward2, term2, _, _ = env2.step(actions)
            
            np.testing.assert_array_equal(obs1_next["grid"], obs2_next["grid"])
            np.testing.assert_array_equal(reward1, reward2)
            assert term1 == term2

