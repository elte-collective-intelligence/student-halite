# Rule Based Bot - Implementation Details

This document provides a detailed explanation of the Rule Based Bot's decision-making process and implementation architecture.

## Overview

The Rule Based Bot operates in **two distinct phases** to determine actions for all cells:

1. **Phase 1: Collect Proposed Moves** - Each cell independently decides its desired action
2. **Phase 2: Resolve Conflicts** - Conflicts between proposed moves are resolved to produce valid final actions

This two-phase approach allows for sophisticated local decision-making while ensuring global consistency.

## Phase 1: Collect Proposed Moves

The first phase iterates through all owned cells and determines what each cell *wants* to do, without considering conflicts. This phase consists of five sequential steps:

### Step 1: Attack or Advance

For each owned cell, the bot checks if it can attack an adjacent enemy or move into a strategically valuable cell.

**Process:**
1. For each direction (NORTH, EAST, SOUTH, WEST), check the neighbor cell
2. Compute two scores:
   - **Enemy Score**: Normalized distance to nearest enemy in that direction (closer = higher score)
   - **Production Score**: Normalized production value at the neighbor (from 3×3 neighborhood normalization)

3. Combine scores using `production_enemy_preference`:
   ```
   combined_score = (preference × production_score) + ((1 - preference) × enemy_score)
   ```
   - `preference = 0.0`: Always prioritize enemies
   - `preference = 1.0`: Always prioritize high production

4. Select the direction with the highest combined score, but only if:
   - The neighbor is not owned by us, OR
   - The neighbor is owned by us but we're stronger (can merge)

**Pre-computation:**
- Enemy distances are pre-computed for all cells and directions to avoid redundant calculations
- Normalized production map is computed once at initialization (3×3 neighborhood sum, then normalized to [0, 1])

### Step 2: Identify Frontier Cells

Frontier cells are cells adjacent to non-owned territory (enemies or neutral cells). These cells form the "border" of our territory.

**Algorithm:**
```python
For each cell (y, x):
    is_frontier[y, x] = False
    For each cardinal neighbor:
        if neighbor is not owned by us:
            is_frontier[y, x] = True
            break
```

### Step 3: Compute Distance to Frontier

Using Breadth-First Search (BFS), compute the shortest distance from each cell to the nearest frontier cell.

**Algorithm:**
1. Initialize all distances to infinity
2. Set frontier cells to distance 0 and add to queue
3. BFS: For each cell, update neighbors with distance + 1
4. Result: `dist_to_frontier[y, x]` = number of steps to nearest border

This creates a gradient map showing how "deep" each cell is within our territory.

### Step 4: Move Inner Cells Toward Frontier

Inner cells (non-frontier cells) attempt to move toward the frontier if they meet the strength threshold.

**Dynamic Strength Threshold:**
```
required_strength = base_strength + scale × (max_distance - cell_distance)
```

Where:
- `base_strength`: Base threshold parameter (default: 50)
- `scale`: Scaling factor (default: 10)
- `max_distance`: Maximum distance to frontier in current state
- `cell_distance`: Distance of this cell to frontier

**Effect:** Cells closer to the frontier need less strength to move, while deep interior cells need much more strength. This creates a pressure gradient that naturally pushes forces toward borders.

**Direction Selection:**
For cells that meet the threshold, find the best direction that:
1. Moves closer to the frontier (decreases `dist_to_frontier`)
2. Maximizes combined score of enemy distance and production (same scoring as Step 1)

### Step 5: Combine Forces for Non-Moving Cells

Cells that don't meet the movement threshold attempt to combine with adjacent neighbors to reach the threshold.

**Process:**
1. For each non-frontier, non-moving cell with insufficient strength:
   - Check all adjacent neighbors
   - Find a neighbor that is:
     - Owned by us
     - Not moving (action = STILL)
     - Not on the frontier
     - Not already combined
     - Has strength > 0
   
2. If combined strength ≥ required_strength and ≤ 255 (max):
   - Move the current cell toward the neighbor
   - Keep neighbor still (it will merge into the current cell)
   - Mark both as "combined" to prevent multiple combinations

**Purpose:** This allows interior cells to pool resources and reach the threshold for movement, enabling coordinated pushes from the interior.

## Phase 2: Resolve Conflicts

After Phase 1, we have proposed actions for each cell. However, these may conflict:

1. **Collision**: Multiple cells try to move to the same destination
2. **Swap**: Two cells try to move to each other's positions (A→B, B→A)
3. **Moving Target**: A cell tries to move to a cell that's also moving

### Conflict Resolution Algorithm

**First Pass: Build Target Map**
- Create a map: `target_map[destination] = list of (source, strength)` tuples
- For each moving cell, record its target destination

**Second Pass: Resolve Conflicts**
For each destination with multiple sources:

1. **Sort sources by strength** (descending), with random tie-breakers for stability
2. **Winner**: The strongest source gets priority
3. **Check for swap**: If the destination cell is also moving toward the winner's source:
   - Break the swap: stronger cell moves, weaker cell waits
   - Mark both as finalized
4. **Check if move is valid**: Can move if destination is neutral OR we're stronger than destination
5. **Apply winner's move**, mark as finalized
6. **All other sources wait** (keep STILL action)

**Third Pass: Finalize Remaining**
- Cells that weren't in any conflict use their proposed action directly

**Result:** Every cell has exactly one valid action, with no collisions or invalid moves.

## Key Mechanisms

### Production Normalization

Production values are normalized using a 3×3 neighborhood approach:

1. For each cell, sum production values in its 3×3 neighborhood (with toroidal wrapping)
2. Normalize all sums to [0, 1] range across the entire map
3. This smooths production values and highlights regions of high production

This normalized production map is computed once at initialization and reused throughout.

### Enemy Distance Computation

For efficiency, enemy distances are pre-computed for all cells and directions:

1. For each cell and each direction:
   - If neighbor is enemy: distance = 1
   - Otherwise: search in that direction (up to `max_search` distance) for first enemy
   - If found: record distance; if not: record `max_search + 1`

2. Normalize all distances to [0, 1] where:
   - 1.0 = closest enemies
   - 0.0 = no enemies found or very far

This pre-computation avoids redundant searches during decision-making.

### Strength Threshold Gradient

The dynamic strength threshold creates a strategic pressure system:

- **Frontier cells** (distance = 0): Need `base_strength` to move
- **One step in** (distance = 1): Need `base_strength + scale` to move
- **Deep interior** (distance = max): Need `base_strength + scale × max_distance` to move

**Benefits:**
- Border cells respond quickly to threats/opportunities
- Interior cells accumulate strength before moving
- Natural flow of forces toward borders
- Prevents premature expansion with weak forces

### Force Combination Strategy

When a cell doesn't meet the threshold, it attempts to combine with one neighbor:

- Only considers non-frontier neighbors (prevents combining border forces)
- Only combines if total strength is sufficient but ≤ 255
- One neighbor donates its strength to the moving cell
- Prevents multiple combinations per cell to avoid complexity

This enables coordinated pushes while maintaining simplicity.

## Parameters

The bot's behavior is controlled by four key parameters:

- **`base_strength`** (default: 50): Base movement threshold for frontier cells
- **`scale`** (default: 10): Multiplier for distance-based threshold scaling
- **`max_search`** (default: 4): Maximum distance to search for enemies
- **`production_enemy_preference`** (default: 0.5): Weight between production (1.0) and enemy (0.0) prioritization

## Decision Tree Summary

```
For each turn:
  Phase 1: Collect Proposed Moves
    ├─ Step 1: Attack adjacent enemies or advance to good neighbors
    │   └─ Score directions by (preference × production + (1-preference) × enemy_distance)
    │
    ├─ Step 2: Identify frontier cells (adjacent to non-owned territory)
    │
    ├─ Step 3: Compute distance-to-frontier map (BFS)
    │
    ├─ Step 4: Move inner cells toward frontier (if strength ≥ threshold)
    │   └─ Threshold = base_strength + scale × (max_dist - cell_dist)
    │
    └─ Step 5: Combine forces for cells below threshold
        └─ Find neighbor to merge with if combined ≥ threshold

  Phase 2: Resolve Conflicts
    ├─ Build target map (destination → list of sources)
    ├─ For each conflict:
    │   ├─ Sort sources by strength
    │   ├─ Detect swaps (A→B, B→A)
    │   ├─ Winner moves, others wait
    │   └─ Break swaps favoring stronger cell
    └─ Finalize remaining non-conflicting moves

  Return final actions array
```

## Implementation Notes

- **Efficiency**: Pre-computation of enemy distances and production normalization avoids redundant calculations
- **Determinism**: Uses random seed for tie-breaking to ensure reproducibility while avoiding bias
- **Torus Wrapping**: All grid operations respect toroidal topology (wraps at edges)
- **Strength Capping**: Combined strengths are capped at 255 (game maximum)
- **Zero-Strength Handling**: Cells with strength = 0 cannot move (must wait and accumulate)

## Code Structure

The implementation is organized into the following methods:

- `__call__()`: Main entry point, orchestrates two phases
- `_collect_proposed_moves()`: Phase 1 coordinator
- `_resolve_conflicts()`: Phase 2 coordinator
- `_attack_or_advance()`: Step 1 implementation
- `_compute_frontier()`: Step 2 implementation
- `_compute_distance_to_frontier()`: Step 3 implementation
- `_move_inner_cells()`: Step 4 implementation
- `_combine_forces_for_non_moving()`: Step 5 implementation
- `_find_best_direction()`: Direction scoring logic
- `_compute_enemy_distances()`: Enemy distance pre-computation
- `_compute_normalized_production()`: Production normalization



