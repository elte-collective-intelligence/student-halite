import os
import uuid
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns


def save_halite_statistics_plot(
    win_rates_by_bot, 
    stats_list_by_bot, 
    global_win_rate,
    player_idx=0, 
    agent_name="Test Agent", 
    save_dir="halite_stats"
):
    """
    Creates a Baseline Evaluation figure with the layout:

        A B
        C D
        E F
        G H

    A = Territory Share
    B = Win Rate (half donut + bar plot)
    C = Production Per Turn
    D = Engagement Efficiency (KDE)
    E = Relative Strength
    F = Cap Losses
    G = Player Damage Dealt
    H = Frontier Length
    """

    # Set modern seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['figure.titlesize'] = 20
    plt.rcParams['figure.titleweight'] = 'bold'

    os.makedirs(save_dir, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/halite_stats_{agent_name}_{timestamp}_{unique_id}.png"

    # Grid: 4 rows x 2 columns
    fig, axes = plt.subplots(4, 2, figsize=(18, 18))
    
    # Add dark header with white title
    header_height = 0.08
    header_rect = plt.Rectangle((0, 1 - header_height), 1, header_height, 
                                transform=fig.transFigure, 
                                facecolor='#2c3e50', edgecolor='none', zorder=10)
    fig.patches.append(header_rect)
    
    # Add white title text in header
    fig.text(0.5, 1 - header_height/2, f"Baseline Evaluation for {agent_name}", 
             fontsize=22, fontweight='bold', color='white',
             ha='center', va='center', zorder=11)

    # Unpack grid
    ax_A, ax_B = axes[0]
    ax_C, ax_D = axes[1]
    ax_E, ax_F = axes[2]
    ax_G, ax_H = axes[3]

    # -----------------------
    # A: Territory Share
    # -----------------------
    # Combine all stats from all bot types
    all_stats = []
    for stats_list in stats_list_by_bot.values():
        all_stats.extend(stats_list)
    
    for stats in all_stats:
        T = stats["territory_share"].shape[1]
        ax_A.plot(np.arange(T), stats["territory_share"][player_idx], alpha=0.3, linewidth=1.5)
    ax_A.set_title("Territory Share", fontsize=16, fontweight='bold')
    ax_A.set_xlabel("Step", fontsize=12)
    ax_A.grid(True, alpha=0.3)

    # -----------------------
    # B: Win Rate (half donut + bar plot)
    # -----------------------
    ax_B.clear()
    ax_B.set_title("Win Rates", fontsize=20)
    ax_B.axis('off')
    
    # Create a GridSpec to split the axis in half
    gs = GridSpec(1, 2, figure=fig, left=ax_B.get_position().x0, 
                  right=ax_B.get_position().x1, 
                  top=ax_B.get_position().y1, 
                  bottom=ax_B.get_position().y0,
                  wspace=0.3)
    
    # Left half: Half donut chart for global win rate (vertical, oriented to the left)
    ax_donut = fig.add_subplot(gs[0, 0])
    ax_donut.set_title("Global", fontsize=16, fontweight='bold', pad=20)
    
    # Create true half donut chart - only 180 degrees, oriented to the left
    # Using custom Wedge patches to create a true semicircle donut
    
    # Calculate angles for half donut (left semicircle: 90 to 270 degrees)
    # Win portion takes up (win_rate * 180) degrees of the semicircle
    win_angle = global_win_rate * 180  # Win takes this many degrees of the 180
    lose_angle = (1 - global_win_rate) * 180  # Lose takes the rest
    
    # Outer and inner radius for donut effect
    outer_radius = 1.0
    inner_radius = 0.5
    width = outer_radius - inner_radius
    
    # Start from 90 degrees (top), go counterclockwise to show left semicircle
    # Win wedge: from 90 to (90 + win_angle)
    win_wedge = mpatches.Wedge((0, 0), outer_radius, 90, 90 + win_angle, 
                               width=width,
                               facecolor='#3498db', edgecolor='white', linewidth=2)
    ax_donut.add_patch(win_wedge)
    
    # Lose wedge: from (90 + win_angle) to 270 (completing the left semicircle)
    lose_wedge = mpatches.Wedge((0, 0), outer_radius, 90 + win_angle, 270,
                                width=width,
                                facecolor='#ecf0f1', edgecolor='white', linewidth=2)
    ax_donut.add_patch(lose_wedge)
    
    # Hide the right half by setting xlim to only show left portion
    ax_donut.set_xlim(-1.2, 0.2)  # Show only left portion
    ax_donut.set_ylim(-1.1, 1.1)   # Show full height
    
    # Add text in center (positioned for left-oriented half donut)
    ax_donut.text(-0.5, 0, f'{global_win_rate:.1%}', 
                  ha='center', va='center', fontsize=28, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#3498db', linewidth=2))
    
    # Hide axes to show only the donut
    ax_donut.set_aspect('equal')
    ax_donut.axis('off')
    
    # Right half: Bar plot for win rates per bot type
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_bar.set_title("", fontsize=16, fontweight='bold', pad=20)
    
    bot_names = list(win_rates_by_bot.keys())
    win_rate_values = [win_rates_by_bot[name] for name in bot_names]
    
    # Create horizontal bar plot with seaborn style
    bars = ax_bar.barh(bot_names, win_rate_values, color='#3498db', alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, win_rate_values):
        ax_bar.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=12,
            fontweight='bold'
        )
    
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Win Rate Against Opponents", fontsize=12, fontweight='bold')
    ax_bar.tick_params(axis='y', labelsize=12)
    ax_bar.tick_params(axis='x', labelsize=11)
    ax_bar.grid(True, axis='x', alpha=0.3, linestyle='--')


    # -----------------------
    # C: Production Per Turn
    # -----------------------
    for stats in all_stats:
        T = stats["production_per_turn"].shape[1]
        ax_C.plot(np.arange(T), stats["production_per_turn"][player_idx], alpha=0.3, linewidth=1.5)
    ax_C.set_title("Relative Production Per Turn", fontsize=16, fontweight='bold')
    ax_C.set_xlabel("Step", fontsize=12)
    ax_C.grid(True, alpha=0.3)

    # -----------------------
    # D: Engagement Efficiency (Scatter: Strength Traded vs Territory Gained)
    # -----------------------
    strength_traded_list = []
    territory_gained_list = []
    for stats in all_stats:
        if player_idx < stats["engagement_efficiency"].shape[0]:
            # engagement_efficiency is now shape (num_players, T, 2) with [strength_traded, territory_gained] per turn
            per_turn_data = stats["engagement_efficiency"][player_idx]  # shape (T, 2)
            strength_traded_turns = per_turn_data[:, 0]  # all strength_traded values for this player
            territory_gained_turns = per_turn_data[:, 1]  # all territory_gained values for this player
            strength_traded_list.extend(strength_traded_turns.tolist())
            territory_gained_list.extend(territory_gained_turns.tolist())
    ax_D.set_title("Engagement Efficiency (Per Turn)", fontsize=16, fontweight='bold')
    if len(strength_traded_list) > 0:
        ax_D.scatter(strength_traded_list, territory_gained_list, alpha=0.6, color='#3498db', s=50, edgecolors='white', linewidths=0.5)
    ax_D.set_xlabel("Strength Traded (per turn)", fontsize=12)
    ax_D.set_ylabel("Territory Gained (per turn)", fontsize=12)
    ax_D.grid(True, alpha=0.3)

    # -----------------------
    # E: Relative Strength
    # -----------------------
    for stats in all_stats:
        T = stats["relative_strength"].shape[1]
        ax_E.plot(np.arange(T), stats["relative_strength"][player_idx], alpha=0.3, linewidth=1.5)
    ax_E.set_title("Relative Strength", fontsize=16, fontweight='bold')
    ax_E.set_xlabel("Step", fontsize=12)
    ax_E.grid(True, alpha=0.3)

    # -----------------------
    # F: Cap Losses
    # -----------------------
    for stats in all_stats:
        T = stats["cap_losses"].shape[1]
        # Remove the last value
        if T > 0:
            ax_F.plot(np.arange(T-1), stats["cap_losses"][player_idx, :-1], alpha=0.3, linewidth=1.5)
    ax_F.set_title("Cap Losses (Cumulative)", fontsize=16, fontweight='bold')
    ax_F.set_xlabel("Step", fontsize=12)
    ax_F.grid(True, alpha=0.3)

    # -----------------------
    # G: Player Damage Dealt
    # -----------------------
    for stats in all_stats:
        T = stats["player_damage_dealt"].shape[1]
        ax_G.plot(np.arange(T), stats["player_damage_dealt"][player_idx], alpha=0.3, linewidth=1.5)
    ax_G.set_title("Player Damage Dealt (Cumulative)", fontsize=16, fontweight='bold')
    ax_G.set_xlabel("Step", fontsize=12)
    ax_G.grid(True, alpha=0.3)

    # -----------------------
    # H: Frontier Length
    # -----------------------
    for stats in all_stats:
        T = stats["frontier_length"].shape[1]
        ax_H.plot(np.arange(T), stats["frontier_length"][player_idx], alpha=0.3, linewidth=1.5)
    ax_H.set_title("Frontier Length", fontsize=16, fontweight='bold')
    ax_H.set_xlabel("Step", fontsize=12)
    ax_H.grid(True, alpha=0.3)

    # Save with adjusted layout to account for header
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
        plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space for header
    plt.savefig(filename, facecolor='white', edgecolor='none')
    plt.close(fig)

    print(f"Baseline Evaluation plot saved to: {filename}")
    return filename

