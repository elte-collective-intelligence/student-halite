#   This script was used to test the Halite environment and agents.
#   It is not used in the training or evaluation pipelines.
#   It is only used to test the environment and agents.

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import sys
import os

# Ensure project root is on sys.path when running as a script
_file_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_file_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.env.env import Halite
from src.agents.agent import Agent
from src.bots.random_bot import RandomBot
from src.bots.rule_based import RuleBasedBot
from src.bots.rule_based_v1 import RuleBasedV1
from src.bots.rule_based_v2 import RuleBasedV2
import click
from rich import print
from rich.panel import Panel
from rich.table import Table
from src.viz.game.GIF_generation.generate_gif import generate_game_gif, generate_border_expansion_heatmap

def create_agents(num_agents: int) -> List[Agent]:
    """Create a list of agents with proper IDs."""

    agents = []
    for i in range(num_agents):
        if i == 0:
            agents.append(RuleBasedBot(agent_id=i))
        elif i == 1:
            agents.append(RuleBasedV1(agent_id=i))
        elif i == 2:
            agents.append(RuleBasedV2(agent_id=i))
        elif i == 3:
            agents.append(RandomBot(agent_id=i))
    return agents

def run_game(
    grid_size: Tuple[int, int],
    num_agents: int,
    seed: int
) -> Tuple[float, dict]:
    """Run a simulation with the specified agents.
    
    Args:
        grid_size: Tuple of (rows, cols) for the grid dimensions
        num_agents: Number of agents in the simulation
        seed: Random seed for reproducibility
    """
    # Initialize environment and agents
    env = Halite(grid_size=grid_size, num_agents=num_agents)
    np.random.seed(seed)
    agents = create_agents(num_agents)
    
    # Track when each agent dies (step count of the last time they were alive)
    death_times = {i: None for i in range(num_agents)}
    
    print()
    print(Panel.fit(
        f"[bold blue]Halite Simulation[/]"
    ))
    print()

    print(
        f"[bold white]Grid size:[/] [blue]{grid_size[0]}x{grid_size[1]}[/]",
        f"[bold white]Number of agents:[/] [blue]{num_agents}[/]",
        f"[bold white]Seed:[/] [blue]{seed}[/]",
        sep="\n"
    )
    print()

    observation, info = env.reset(seed=seed)
    state = env.state  # Access the internal state
    
    # Storage for trajectory
    states, actions = [], []
    
    # Main simulation loop
    terminated = False
    while not terminated:
        print(f"[bold white]Turn[/] [blue]{state.step_count}[/]", end="\r")
        
        # Check for agents that just died
        for agent_id in range(num_agents):
            if death_times[agent_id] is None and not state.alive[agent_id]:
                death_times[agent_id] = state.step_count - 1
        
        # Generate actions for each agent
        # Use a per-step seed derived from base seed and step count for deterministic behavior
        step_seed = seed + state.step_count if seed is not None else None
        current_actions = []
        for agent in agents:
            agent_action = agent(observation["grid"], seed=step_seed)
            current_actions.append(agent_action)
        
        # Stack actions [num_agents, height, width]
        action = np.stack(current_actions)
        
        # Store and step
        states.append(state)
        actions.append(action)
        observation, reward, terminated, truncated, info = env.step(action)
        state = env.state  # Update state reference
    
    # Final check for agents that might have died in the last step
    for agent_id in range(num_agents):
        if death_times[agent_id] is None and not state.alive[agent_id]:
            death_times[agent_id] = state.step_count - 1
    
    print(f"[white]Turn[/] [bold]{state.step_count}[/]\n")

    # Finalization
    states.append(state)
    
    # Calculate final territory counts
    ownership_grid = state.grid[0]  # Ownership channel
    territory_counts = {
        agent_id: np.sum(ownership_grid == (agent_id + 1))  # +1 because env uses 1-based IDs
        for agent_id in range(num_agents)
    }
    
    # Calculate rankings
    max_step = state.step_count
    
    # Create list of (agent_id, death_time, territory_count) tuples
    agent_stats = []
    for agent_id in range(num_agents):
        death_time = death_times[agent_id] if death_times[agent_id] is not None else max_step
        agent_stats.append((agent_id, death_time, territory_counts[agent_id]))
    
    # Sort by:
    # 1. Death time (descending - later death is better)
    # 2. Territory count (descending - more territory is better)
    ranked_agents = sorted(
        agent_stats,
        key=lambda x: (-x[1], -x[2])  # Sort by death_time desc, then territory desc
    )
    
    rankings = {agent_id: rank+1 for rank, (agent_id, _, _) in enumerate(ranked_agents)}
    
    # Create results table
    results_table = Table(show_header=True, header_style="bold blue")
    results_table.add_column("Rank", style="dim", justify="right")
    results_table.add_column("Agent ID", justify="center")
    results_table.add_column("Last Alive Turn", justify="right")
    results_table.add_column("Territory", justify="right")
    results_table.add_column("Status", justify="center")
    
    for rank, (agent_id, death_time, territory) in enumerate(ranked_agents, 1):
        status = "Alive" if death_time == max_step and territory > 0 else "Eliminated"
        results_table.add_row(
            str(rank),
            str(agent_id),
            str(death_time),
            str(territory),
            f"[green]{status}[/]" if status == "Alive" else f"[red]{status}[/]"
        )

    print(results_table)
    
    env.save(states, actions)
    generate_border_expansion_heatmap(states, output_path="border_expansion_heatmap.png", num_players=num_agents, cell_size=30, colormap='hot', player_names=[bot.name for bot in agents])
    generate_game_gif(states, actions, output_path="halite_game.gif", fps=4, cell_size=30, show_strength_heatmap=True, show_expansion_fronts=False, alpha_strength=0.6, alpha_front=0.8, num_players=num_agents, player_names=[bot.name for bot in agents])

def validate_grid_size(ctx, param, value):
    """Click callback to validate and parse grid size from space-separated values."""
    try:
        parts = value.split()
        if len(parts) == 1:
            size = int(parts[0])
            return (size, size)
        elif len(parts) == 2:
            return tuple(map(int, parts))
        else:
            raise ValueError
    except (ValueError, AttributeError):
        raise click.BadParameter('Grid size must be either a single integer or two space-separated integers (e.g., "8" or "8 10")')

@click.command()
@click.option('--grid-size', 
              default='35', 
              callback=validate_grid_size,
              help='Grid dimensions as "rows cols" or single number for square grid (e.g., "8" or "8 10")')
@click.option('--num-agents', default=4, help='Number of agents in the simulation', type=int)
@click.option('--seed', default=0, help='Random seed for reproducibility', type=int)
def main(grid_size: Tuple[int, int], num_agents: int, seed: int):
    """Run a Halite simulation with the specified parameters."""
    run_game(
        grid_size=grid_size,
        num_agents=num_agents,
        seed=seed
    )

if __name__ == "__main__":
    main()