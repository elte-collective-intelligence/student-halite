import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime
import uuid

def create_grid_search_heatmap(
    win_rate_matrix: np.ndarray,
    base_strength_range: list[int],
    production_enemy_preference_range: list[float],
    episodes: int,
    save_dir: str
) -> str:    
    """
    Create and save a heatmap visualization of the grid search results.
    
    Parameters
    ----------
    win_rate_matrix : np.ndarray
        Matrix of win rates
    base_strength_range : List[int]
        List of base_strength values
    production_enemy_preference_range : List[float]
        List of production_enemy_preference values (0 = always enemies, 1 = always production)
    episodes : int
        Number of episodes used for evaluation
    save_dir : str
        Directory to save the heatmap
        
    Returns
    -------
    str
        Path to the saved heatmap image
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set modern style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size': 11,
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#2c3e50',
        'axes.labelcolor': '#2c3e50',
        'xtick.color': '#34495e',
        'ytick.color': '#34495e',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })
    
    # Create figure with better aspect ratio
    fig = plt.figure(figsize=(14, 10), facecolor='white')
    
    # Create main axes with padding (more space at bottom for stats)
    ax = fig.add_axes([0.1, 0.18, 0.75, 0.70])
    
    # Create heatmap using matplotlib imshow with modern colormap
    # Using 'viridis' - perceptually uniform, colorblind-friendly, modern
    im = ax.imshow(
        win_rate_matrix,
        cmap='viridis',
        aspect='auto',
        vmin=0.0,
        vmax=1.0,
        interpolation='nearest',
        alpha=0.9
    )
    
    # Add subtle grid lines for cell separation
    ax.set_xticks(np.arange(len(production_enemy_preference_range) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(base_strength_range) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2.5, alpha=0.8)
    ax.tick_params(which="minor", size=0)
    
    # Set tick labels with better formatting
    ax.set_xticks(np.arange(len(production_enemy_preference_range)))
    ax.set_yticks(np.arange(len(base_strength_range)))
    ax.set_xticklabels(production_enemy_preference_range, fontsize=12, color='#2c3e50')
    ax.set_yticklabels(base_strength_range, fontsize=12, color='#2c3e50')
    
    # Add text annotations with better contrast and styling
    for i in range(len(base_strength_range)):
        for j in range(len(production_enemy_preference_range)):
            value = win_rate_matrix[i, j]
            # Adjust threshold for viridis colormap (darker at low values, brighter at high)
            text_color = "white" if value < 0.5 else "#2c3e50"
            
            # Add subtle text shadow for better readability
            ax.text(
                j, i, f'{value:.2f}',
                ha="center", va="center",
                color=text_color,
                fontsize=11,
                fontweight='600',
                family='monospace'
            )
    
    # Highlight best cell with prominent black border
    best_idx = np.unravel_index(np.argmax(win_rate_matrix), win_rate_matrix.shape)
    
    # Outer border (thicker)
    best_rect_outer = Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5),
        1, 1,
        linewidth=5,
        edgecolor='black',
        facecolor='none',
        zorder=11
    )
    ax.add_patch(best_rect_outer)
    
    # Inner border (thinner for depth)
    best_rect_inner = Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5),
        1, 1,
        linewidth=2.5,
        edgecolor='black',
        facecolor='none',
        zorder=12
    )
    ax.add_patch(best_rect_inner)
    
    # Modern colorbar with better styling
    cbar_ax = fig.add_axes([0.87, 0.18, 0.02, 0.70])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(
        'Win Rate',
        fontsize=13,
        fontweight='600',
        color='#2c3e50',
        labelpad=15
    )
    cbar.ax.tick_params(labelsize=11, color='#34495e')
    cbar.outline.set_edgecolor('#bdc3c7')
    cbar.outline.set_linewidth(1)
    
    # Set labels with better styling
    ax.set_xlabel(
        'Production Enemy Preference',
        fontsize=14,
        fontweight='600',
        color='#2c3e50',
        labelpad=12
    )
    ax.set_ylabel(
        'Base Strength',
        fontsize=14,
        fontweight='600',
        color='#2c3e50',
        labelpad=12
    )
    
    # Modern title with better spacing
    ax.set_title(
        f'RuleBasedBot2 Grid Search Results',
        fontsize=18,
        fontweight='700',
        color='#2c3e50',
        pad=25
    )
    ax.text(
        0.5, 1.08,
        f'Win Rate Heatmap ({episodes} episodes per combination)',
        transform=ax.transAxes,
        ha='center',
        fontsize=13,
        color='#7f8c8d',
        style='italic'
    )
    
    # Calculate statistics
    max_win_rate = np.max(win_rate_matrix)
    min_win_rate = np.min(win_rate_matrix)
    mean_win_rate = np.mean(win_rate_matrix)
    std_win_rate = np.std(win_rate_matrix)
    
    # Convert numpy int64 to Python int for indexing (needed for OmegaConf ListConfig)
    best_base_strength = base_strength_range[int(best_idx[0])]
    best_production_enemy_preference = production_enemy_preference_range[int(best_idx[1])]
    
    # Modern statistics box with better styling
    stats_text = (
        f'Best Parameters: base_strength={best_base_strength}, production_enemy_preference={best_production_enemy_preference} '
        f'(Win Rate: {max_win_rate:.3f})'
    )
    
    stats_box = fig.add_axes([0.1, 0.02, 0.8, 0.08])
    stats_box.axis('off')
    stats_box.text(
        0.5, 0.7,
        stats_text,
        transform=stats_box.transAxes,
        ha='center',
        va='center',
        fontsize=12,
        fontweight='600',
        color='#2c3e50',
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='#ecf0f1',
            edgecolor='#bdc3c7',
            linewidth=1.5,
            alpha=0.9
        )
    )
    
    # Additional stats in smaller text
    stats_box.text(
        0.5, 0.25,
        f'Mean: {mean_win_rate:.3f}  |  Std: {std_win_rate:.3f}  |  '
        f'Min: {min_win_rate:.3f}  |  Max: {max_win_rate:.3f}',
        transform=stats_box.transAxes,
        ha='center',
        va='center',
        fontsize=10,
        color='#7f8c8d'
    )
    
    # Save figure with high quality
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{save_dir}/grid_search_rule_based_bot2_{timestamp}_{unique_id}.png"
    fig.savefig(
        filename,
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        pad_inches=0.2
    )
    plt.close(fig)
    
    print(f"\nHeatmap saved to: {filename}")
    return filename