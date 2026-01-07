#!/usr/bin/env python3
"""
Script to generate replay GIF and heatmap from a .hlt file.

Usage:
    python src/viz/game/visualize_hlt.py <path_to_hlt_file> [--output-dir <dir>] [--fps <fps>]
    # Docker one-liner:
    ./docker/docker-run.sh python src/viz/game/visualize_hlt.py <path_to_hlt_file> [--output-dir <dir>] [--fps <fps>]
"""

import json
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.env.types import State
from src.env.utils_env import _get_action_mask
from src.viz.game.GIF_generation.generate_gif import (
    generate_game_gif,
    generate_border_expansion_heatmap
)


def load_hlt_file(filepath: str) -> dict:
    """
    Load a .hlt replay file.
    
    Args:
        filepath: Path to the .hlt file
        
    Returns:
        Dictionary containing the replay data
        
    Raises:
        ValueError: If required fields are missing
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Validate required fields
    required_fields = ['version', 'width', 'height', 'num_players', 'num_frames', 'productions', 'frames']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields in .hlt file: {missing_fields}")
    
    return data


def hlt_to_states(hlt_data: dict) -> List[State]:
    """
    Convert .hlt file data to a list of State objects.
    
    Args:
        hlt_data: Dictionary loaded from .hlt file
        
    Returns:
        List of State objects
    """
    width = hlt_data['width']
    height = hlt_data['height']
    num_players = hlt_data['num_players']
    num_frames = hlt_data['num_frames']
    productions = np.array(hlt_data['productions'], dtype=np.int32)
    frames = hlt_data['frames']
    
    states = []
    
    for t, frame in enumerate(frames):
        # Each frame is a list of [owner, strength] pairs
        # Convert to numpy arrays
        frame_array = np.array(frame, dtype=np.int32)  # (H, W, 2)
        
        # Extract owner and strength
        owner = frame_array[:, :, 0]  # (H, W)
        strength = frame_array[:, :, 1]  # (H, W)
        
        # Production is constant throughout the game (from first state)
        # Stack to create grid: (3, H, W)
        grid = np.stack([owner, strength, productions], axis=0)
        
        # Create action mask
        action_mask = _get_action_mask(grid, num_players)
        
        # Create alive array (all players alive, since .hlt doesn't track this)
        alive = np.ones((num_players,), dtype=bool)
        
        # Create State object
        state = State(
            grid=grid,
            step_count=t,
            alive=alive,
            action_mask=action_mask
        )
        
        states.append(state)
    
    return states


def main():
    parser = argparse.ArgumentParser(
        description='Generate replay GIF and heatmap from a .hlt file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/viz/game/visualize_hlt.py outputs/games/mappo_98137A/ep18000.hlt
  python src/viz/game/visualize_hlt.py game.hlt --output-dir outputs/visualizations --fps 3
        """
    )
    parser.add_argument(
        'hlt_file',
        type=str,
        help='Path to the .hlt replay file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save output files (default: same directory as .hlt file)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=4,
        help='Frames per second for the GIF (default: 4)'
    )
    parser.add_argument(
        '--cell-size',
        type=int,
        default=30,
        help='Size of each cell in pixels (default: 30)'
    )
    parser.add_argument(
        '--no-strength-heatmap',
        action='store_true',
        help='Disable strength heatmap overlay in GIF'
    )
    parser.add_argument(
        '--expansion-fronts',
        action='store_true',
        help='Enable expansion fronts in GIF (default: False)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.hlt_file):
        print(f"Error: File not found: {args.hlt_file}")
        sys.exit(1)
    
    if not args.hlt_file.endswith('.hlt'):
        print(f"Warning: File does not have .hlt extension: {args.hlt_file}")
    
    # Load .hlt file
    print(f"Loading .hlt file: {args.hlt_file}")
    try:
        hlt_data = load_hlt_file(args.hlt_file)
    except Exception as e:
        print(f"Error loading .hlt file: {e}")
        sys.exit(1)
    
    # Convert to states
    print("Converting to State objects...")
    try:
        states = hlt_to_states(hlt_data)
    except Exception as e:
        print(f"Error converting to states: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Loaded {len(states)} frames")
    print(f"Grid size: {hlt_data['width']}x{hlt_data['height']}")
    print(f"Number of players: {hlt_data['num_players']}")
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(args.hlt_file))
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename (without extension) for game_id
    base_name = os.path.splitext(os.path.basename(args.hlt_file))[0]
    
    # Generate GIF
    gif_path = os.path.join(output_dir, "game.gif")
    print(f"\nGenerating GIF: {gif_path}")
    try:
        generate_game_gif(
            states=states,
            actions=None,  # Actions not needed for visualization
            output_path=gif_path,
            fps=args.fps,
            cell_size=args.cell_size,
            show_strength_heatmap=not args.no_strength_heatmap,
            show_expansion_fronts=args.expansion_fronts,
            alpha_strength=0.6,
            alpha_front=0.8,
            num_players=hlt_data['num_players'],
            player_names=hlt_data.get('player_names'),
            game_id=base_name
        )
        print(f"✓ GIF saved to: {gif_path}")
    except Exception as e:
        print(f"Error generating GIF: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate heatmap
    heatmap_path = os.path.join(output_dir, f"{base_name}_heatmap.png")
    print(f"\nGenerating heatmap: {heatmap_path}")
    try:
        generate_border_expansion_heatmap(
            states=states,
            output_path=heatmap_path,
            num_players=hlt_data['num_players'],
            cell_size=args.cell_size,
            colormap='hot',
            player_names=hlt_data.get('player_names'),
            game_id=base_name
        )
        print(f"✓ Heatmap saved to: {heatmap_path}")
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✓ Visualization complete!")
    print(f"  GIF: {gif_path}")
    print(f"  Heatmap: {heatmap_path}")


if __name__ == '__main__':
    main()

