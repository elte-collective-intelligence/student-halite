"""
GIF Generation Module for Halite Game Visualization

This module provides functionality to generate animated GIFs from game states and actions,
showing ownership, strength heatmaps, and expansion fronts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Optional, Tuple
import imageio
import os
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from src.env.types import State


def get_expansion_fronts(owner_map: np.ndarray, player_id: int, height: int, width: int) -> np.ndarray:
    """
    Calculate expansion front cells for a given player.
    An expansion front cell is owned by the player and has at least one neighbor
    that is not owned by that player.
    
    Args:
        owner_map: 2D array of shape (height, width) with ownership values
        player_id: Player ID (1-based)
        height: Grid height
        width: Grid width
        
    Returns:
        Boolean array of shape (height, width) indicating expansion front cells
    """
    front_mask = np.zeros((height, width), dtype=bool)
    
    for r in range(height):
        for c in range(width):
            if owner_map[r, c] != player_id:
                continue
            # Check neighbors
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = (r + dr) % height
                cc = (c + dc) % width
                if owner_map[rr, cc] != player_id:
                    front_mask[r, c] = True
                    break
    
    return front_mask


def create_frame(
    state: State,
    height: int,
    width: int,
    num_players: int,
    cell_size: int = 30,
    show_strength_heatmap: bool = True,
    show_expansion_fronts: bool = True,
    alpha_strength: float = 0.6,
    alpha_front: float = 0.8,
    player_names: Optional[List[str]] = None,
    game_id: Optional[str] = None
) -> np.ndarray:
    """
    Create a single frame visualization of the game state.
    
    Args:
        state: State object containing grid information
        height: Grid height
        width: Grid width
        num_players: Number of players
        cell_size: Size of each cell in pixels
        show_strength_heatmap: Whether to overlay strength heatmap
        show_expansion_fronts: Whether to highlight expansion fronts
        alpha_strength: Transparency for strength heatmap (0-1)
        alpha_front: Transparency for expansion front overlay (0-1)
        
    Returns:
        RGB array of the frame
    """
    owner = state.grid[0]  # (height, width)
    strength = state.grid[1]  # (height, width)
    production = state.grid[2]  # (height, width)
    
    # Dark gray background (darker)
    dark_gray = (0.15, 0.15, 0.15)
    
    # Create figure with space for legend on the right
    # Increase legend width (6 units)
    legend_width = 6.0
    fig_width = (width + legend_width) * cell_size / 100
    fig_height = height * cell_size / 100
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor=dark_gray)
    fig.patch.set_facecolor(dark_gray)
    
    # Create main axes for the grid (positioned to the left, leaving space for legend on the right)
    ax = fig.add_axes([0, 0, width / (width + legend_width), 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.invert_yaxis()  # Match typical grid visualization (top-left origin)
    ax.set_facecolor(dark_gray)
    
    # Remove all spines and borders
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Define colors for players (excluding 0 which is neutral/unowned)
    # Using distinct, perceptually different colors
    player_colors = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.2, 0.8),  # Blue
        (0.2, 0.8, 0.2),  # Green
        (0.8, 0.8, 0.2),  # Yellow
        (0.8, 0.2, 0.8),  # Magenta
        (0.2, 0.8, 0.8),  # Cyan
        (0.6, 0.3, 0.1),  # Brown
        (0.3, 0.1, 0.6),  # Purple
    ]
    
    # Compute production range for opacity calculation
    # Reduced contrast: Max production -> opacity = 0.75, Min production -> opacity = 0.95
    prod_min = float(production.min())
    prod_max = float(production.max())
    prod_range = prod_max - prod_min if prod_max > prod_min else 1.0
    
    # Normalize strength (0-255 range)
    strength_normalized = strength / 255.0 if strength.max() > 0 else np.zeros_like(strength)
    
    # Draw cells
    for r in range(height):
        for c in range(width):
            owner_id = int(owner[r, c])
            prod_val = float(production[r, c])
            strength_val = float(strength[r, c])
            
            # Base color based on ownership
            if owner_id == 0:
                # Neutral/unowned cells - light gray
                cell_color = (0.7, 0.7, 0.7)
            else:
                # Player-owned cells - use player color
                color_idx = (owner_id - 1) % len(player_colors)
                cell_color = player_colors[color_idx]
            
            # Calculate cell background opacity based on production
            # Reduced contrast: Max production -> opacity = 0.75, Min production -> opacity = 0.95
            if prod_range > 0:
                # Linear interpolation: opacity goes from 0.95 (min) to 0.75 (max)
                cell_opacity = 0.95 - 0.2 * ((prod_val - prod_min) / prod_range)
            else:
                cell_opacity = 0.85  # Default if all production is same
            
            # Create rectangle for cell background with production-based opacity
            rect = Rectangle((c, r), 1, 1, 
                           facecolor=cell_color, 
                           edgecolor='none',  # No borders
                           linewidth=0,
                           alpha=cell_opacity)
            ax.add_patch(rect)
            
            # Draw inner square for strength (for all cells, including neutral)
            if strength_val > 0:
                # Calculate inner square size based on strength
                # Max strength (255) -> 90% of cell, scaled linearly
                max_inner_size = 0.9  # 90% of cell
                inner_size = max_inner_size * strength_normalized[r, c]
                
                # Center the inner square
                cell_size_units = 1.0  # Each cell is 1 unit
                inner_half = inner_size / 2.0
                center_x = c + 0.5
                center_y = r + 0.5
                
                # Draw inner square with full opacity (1.0) and same color as cell
                inner_rect = Rectangle(
                    (center_x - inner_half, center_y - inner_half),
                    inner_size, inner_size,
                    facecolor=cell_color,
                    edgecolor='white' if strength_val >= 255.0 else 'none',
                    linewidth=1.5 if strength_val >= 255.0 else 0,
                    alpha=1.0  # Fully opaque
                )
                ax.add_patch(inner_rect)
    
    # Draw expansion fronts if enabled
    if show_expansion_fronts:
        for player_id in range(1, num_players + 1):
            front_mask = get_expansion_fronts(owner, player_id, height, width)
            color_idx = (player_id - 1) % len(player_colors)
            front_color = player_colors[color_idx]
            
            for r in range(height):
                for c in range(width):
                    if front_mask[r, c]:
                        # Draw a border/outline on expansion front cells
                        front_rect = Rectangle((c, r), 1, 1,
                                             facecolor='none',
                                             edgecolor=front_color,
                                             linewidth=2.5,
                                             alpha=alpha_front)
                        ax.add_patch(front_rect)
    
    # Create legend on the right side
    # Position legend axis on the right
    legend_left = width / (width + legend_width) + 0.01
    legend_width_frac = (legend_width / (width + legend_width)) * 0.95
    legend_ax = fig.add_axes([legend_left, 0.05, legend_width_frac, 0.9])
    legend_ax.axis('off')
    legend_ax.set_facecolor(dark_gray)
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    
    # Create legend patches
    legend_elements = []
    for i in range(num_players):
        player_id = i + 1
        color_idx = i % len(player_colors)
        color = player_colors[color_idx]
        # Use player name if provided, otherwise default to "Player {id}"
        if player_names is not None and i < len(player_names):
            player_label = player_names[i]
        else:
            player_label = f'Player {player_id}'
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none', label=player_label)
        )
    
    # Add neutral cell to legend
    legend_elements.append(
        plt.Rectangle((0, 0), 1, 1, facecolor=(0.7, 0.7, 0.7), edgecolor='none', label='Neutral')
    )
    
    # Create legend
    legend = legend_ax.legend(handles=legend_elements, loc='upper left', 
                             frameon=False, fontsize=max(8, cell_size // 4), 
                             labelcolor='white', handlelength=1.5, handletextpad=0.5)
    
    # Add game ID in bottom left corner if provided
    if game_id is not None:
        ax.text(0, height, f'Game ID: {game_id}', 
               fontsize=max(8, cell_size // 5), color='white',
               verticalalignment='bottom', horizontalalignment='left',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'))
    
    # Convert to numpy array using BytesIO (more compatible across matplotlib versions)
    # Use fixed bbox to ensure consistent frame sizes for GIF
    # Use dark gray background, no padding to avoid black borders
    buf = BytesIO()
    # Save with no padding and ensure figure edge matches background
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, 
                dpi=100, facecolor=dark_gray, edgecolor='none', 
                transparent=False)
    buf.seek(0)
    img = Image.open(buf)
    
    # Convert to RGB if needed (remove alpha channel, use dark gray background)
    dark_gray_rgb = tuple(int(c * 255) for c in dark_gray)
    
    if img.mode == 'RGBA':
        # Create dark gray background and paste the image
        rgb_img = Image.new('RGB', img.size, dark_gray_rgb)
        rgb_img.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img_array = np.array(rgb_img)
    else:
        img_array = np.array(img.convert('RGB'))
    
    # Ensure background is consistently dark gray (replace any near-black borders)
    # This handles cases where bbox_inches='tight' might add slight borders
    mask = np.all(img_array < 20, axis=2)  # Very dark pixels (near black)
    img_array[mask] = dark_gray_rgb
    
    plt.close(fig)
    buf.close()
    
    return img_array


def generate_border_expansion_heatmap(
    states: List[State],
    output_path: str = "border_expansion_heatmap.png",
    num_players: Optional[int] = None,
    cell_size: int = 30,
    colormap: str = 'hot',
    player_names: Optional[List[str]] = None,
    game_id: Optional[str] = None
) -> str:
    """
    Generate a heatmap showing border expansion activity throughout the game.
    Each cell is colored based on how many times it was on a border/frontier during the game.
    
    Args:
        states: List of State objects representing the game states
        output_path: Path where the heatmap will be saved
        num_players: Number of players (inferred from first state if not provided)
        cell_size: Size of each cell in pixels (controls resolution)
        colormap: Matplotlib colormap name to use for the heatmap (default: 'hot')
        player_names: Optional list of player names (will default to "Player 1", "Player 2", etc. if not provided)
        game_id: Optional game ID to display in the title
        
    Returns:
        Path to the saved heatmap image
        
    Raises:
        ValueError: If inputs are invalid
        
    Example:
        >>> from src.viz.game.GIF_generation import generate_border_expansion_heatmap
        >>> heatmap_path = generate_border_expansion_heatmap(states, output_path="border_heatmap.png")
    """
    # Validate inputs
    if states is None or len(states) == 0:
        raise ValueError("States list cannot be empty")
    
    # Get grid dimensions
    first_state_grid = states[0].grid
    height, width = first_state_grid.shape[1:]
    
    # Infer number of players if not provided
    if num_players is None:
        max_owner = int(first_state_grid[0].max())
        num_players = max(1, max_owner)
    
    # Validate player_names if provided
    if player_names is not None and len(player_names) < num_players:
        raise ValueError(
            f"player_names list has {len(player_names)} names but {num_players} players are present"
        )
    
    # Initialize heatmap counter (counts how many times each cell was on a border)
    border_count = np.zeros((height, width), dtype=np.int32)
    
    # Process all states to count border occurrences
    print("Computing border expansion counts...")
    for state in tqdm(states, desc="Processing states", unit="state"):
        owner = state.grid[0]
        
        # For each player, find expansion fronts and increment counters
        for player_id in range(1, num_players + 1):
            front_mask = get_expansion_fronts(owner, player_id, height, width)
            border_count[front_mask] += 1
    
    # Get spawn points from the first state (initial player positions)
    first_state = states[0]
    initial_owner = first_state.grid[0]
    
    # Define player colors (same as in GIF generation)
    player_colors = [
        (0.8, 0.2, 0.2),  # Red
        (0.2, 0.2, 0.8),  # Blue
        (0.2, 0.8, 0.2),  # Green
        (0.8, 0.8, 0.2),  # Yellow
        (0.8, 0.2, 0.8),  # Magenta
        (0.2, 0.8, 0.8),  # Cyan
        (0.6, 0.3, 0.1),  # Brown
        (0.3, 0.1, 0.6),  # Purple
    ]
    
    # Create the heatmap visualization with white/transparent background
    # Add extra width for the legend on the right
    legend_width_units = 3.0
    fig_width = (width + legend_width_units) * cell_size / 100
    fig_height = height * cell_size / 100
    
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=100, facecolor='white')
    fig.patch.set_facecolor('white')
    
    # Create main axes for the heatmap (left side, leaving space for legend on right)
    ax = fig.add_axes([0, 0, width / (width + legend_width_units), 1])
    ax.set_facecolor('white')
    
    # Create heatmap using imshow
    im = ax.imshow(border_count, cmap=colormap, interpolation='nearest', origin='upper')
    
    # Highlight spawn points with colored borders
    for player_id in range(1, num_players + 1):
        spawn_mask = (initial_owner == player_id)
        color_idx = (player_id - 1) % len(player_colors)
        spawn_color = player_colors[color_idx]
        
        # Draw borders around spawn point cells using Rectangle patches
        for r in range(height):
            for c in range(width):
                if spawn_mask[r, c]:
                    # Draw a rectangle border around the spawn cell
                    spawn_border = Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor='none',
                        edgecolor=spawn_color,
                        linewidth=3.0,
                        alpha=1.0
                    )
                    ax.add_patch(spawn_border)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Border Expansion Count', rotation=270, labelpad=20, color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    cbar.ax.yaxis.label.set_color('black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
    
    # Set labels and title
    title = 'Border Expansion Heatmap'
    if game_id is not None:
        title = f'Border Expansion Heatmap ({game_id})'
    ax.set_title(title, color='black', fontsize=14, pad=10)
    ax.set_xlabel('Width', color='black')
    ax.set_ylabel('Height', color='black')
    
    # Style the axes
    ax.tick_params(colors='black')
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    
    # Create spawnpoints legend outside the heatmap
    legend_elements = []
    for i in range(num_players):
        player_id = i + 1
        color_idx = i % len(player_colors)
        color = player_colors[color_idx]
        # Use player name if provided, otherwise default
        if player_names is not None and i < len(player_names):
            player_label = player_names[i]
        else:
            player_label = f'Player {player_id}'
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor='none', label=player_label)
        )
    
    # Create legend axis on the right side
    legend_left = width / (width + legend_width_units) + 0.1
    legend_width_frac = (legend_width_units / (width + legend_width_units)) * 0.9
    legend_ax = fig.add_axes([legend_left, 0.1, legend_width_frac, 0.8])
    legend_ax.axis('off')
    legend_ax.set_facecolor('white')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    
    # Create modern, clean legend
    spawn_legend = legend_ax.legend(handles=legend_elements, loc='upper left', 
                                   frameon=False, fontsize=11, 
                                   title='Spawnpoints', title_fontsize=13,
                                   handlelength=1.2, handletextpad=0.6,
                                   columnspacing=0.8, labelspacing=0.5)
    
    # Style the legend title and text
    title = spawn_legend.get_title()
    title.set_color('#2C3E50')
    title.set_weight('bold')
    for text in spawn_legend.get_texts():
        text.set_color('#34495E')
        text.set_weight('normal')
    
    # Remove ticks for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the heatmap
    plt.tight_layout()
    plt.savefig(output_path, facecolor='white', bbox_inches='tight', dpi=150, transparent=False)
    plt.close(fig)
    
    print(f"Border expansion heatmap saved to: {output_path}")
    return output_path


def generate_game_gif(
    states: List[State],
    actions: Optional[List[np.ndarray]] = None,
    output_path: str = "halite_game.gif",
    fps: int = 2,
    cell_size: int = 30,
    show_strength_heatmap: bool = True,
    show_expansion_fronts: bool = True,
    alpha_strength: float = 0.6,
    alpha_front: float = 0.8,
    num_players: Optional[int] = None,
    player_names: Optional[List[str]] = None,
    game_id: Optional[str] = None
) -> str:
    """
    Generate an animated GIF from a sequence of game states and actions.
    
    This function creates a GIF visualization showing:
    - Grid cells colored by ownership (each player has a distinct color)
    - Strength heatmap overlay (brighter = higher strength)
    - Expansion fronts highlighted with colored borders (cells adjacent to enemy/neutral territory)
    
    Args:
        states: List of State objects representing the game states
        actions: Optional list of action arrays taken between states (shape: (num_players, H, W))
                 Note: If provided, len(actions) should be len(states) - 1. Actions are accepted
                 for API consistency with env.save() but are not used in the visualization.
        output_path: Path where the GIF will be saved
        fps: Frames per second for the GIF
        cell_size: Size of each cell in pixels (controls resolution)
        show_strength_heatmap: Whether to overlay strength heatmap
        show_expansion_fronts: Whether to highlight expansion fronts
        alpha_strength: Transparency for strength heatmap (0-1)
        alpha_front: Transparency for expansion front overlay (0-1)
        num_players: Number of players (inferred from first state if not provided)
        player_names: Optional list of player names (will default to "Player 1", "Player 2", etc. if not provided)
        game_id: Optional game ID to display in the bottom left corner
        
    Returns:
        Path to the saved GIF file
        
    Raises:
        ValueError: If inputs are invalid
        
    Example:
        >>> from src.viz.game.GIF_generation import generate_game_gif
        >>> # After running a game and collecting states and actions
        >>> gif_path = generate_game_gif(states, actions, output_path="my_game.gif", fps=3)
    """
    # Validate inputs
    if states is None or len(states) == 0:
        raise ValueError("States list cannot be empty")
    
    if actions is not None and len(actions) != len(states) - 1:
        raise ValueError(
            f"Number of actions ({len(actions)}) "
            f"should be one less than number of states ({len(states)})"
        )
    
    # Get grid dimensions
    first_state_grid = states[0].grid
    height, width = first_state_grid.shape[1:]
    
    # Infer number of players if not provided
    if num_players is None:
        max_owner = int(first_state_grid[0].max())
        num_players = max(1, max_owner)  # At least 1, or max owner ID
    
    # Validate player_names if provided
    if player_names is not None and len(player_names) < num_players:
        raise ValueError(
            f"player_names list has {len(player_names)} names but {num_players} players are present"
        )
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Generate frames with progress bar
    frames = []
    target_size = None
    for i, state in enumerate(tqdm(states, desc="Generating GIF frames", unit="frame")):
        frame = create_frame(
            state=state,
            height=height,
            width=width,
            num_players=num_players,
            cell_size=cell_size,
            show_strength_heatmap=show_strength_heatmap,
            show_expansion_fronts=show_expansion_fronts,
            alpha_strength=alpha_strength,
            alpha_front=alpha_front,
            player_names=player_names,
            game_id=game_id
        )
        
        # Normalize frame size to ensure all frames are the same dimensions
        if target_size is None:
            target_size = (frame.shape[1], frame.shape[0])  # (width, height)
        else:
            # Resize frame to match first frame size
            if frame.shape[:2] != target_size[::-1]:  # target_size is (w, h), frame.shape is (h, w, 3)
                img = Image.fromarray(frame)
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                frame = np.array(img)
        
        frames.append(frame)
    
    # Save as GIF
    print("Saving GIF...")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"GIF saved to: {output_path}")
    
    return output_path
