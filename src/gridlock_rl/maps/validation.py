import numpy as np
from collections import deque
from gridlock_rl.core.constants import TileType, Action

def get_neighbors(pos, grid_shape):
    """Return valid neighbors for a given position."""
    r, c = pos
    h, w = grid_shape
    # Up, Right, Down, Left
    deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    neighbors = []
    for dr, dc in deltas:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            neighbors.append((nr, nc))
    return neighbors

def is_reachable(grid, start_pos, target_pos_set):
    """
    Check if ANY of the target positions are reachable from start_pos.
    Returns the set of reached targets.
    Traverses EMPTY, START, GOAL, KEY.
    Blocks on WALL, TRAP.
    """
    h, w = grid.shape
    visited = set()
    queue = deque([start_pos])
    visited.add(start_pos)
    
    reached_targets = set()
    target_pos_set = set(target_pos_set) # optimize lookup

    while queue:
        curr = queue.popleft()
        
        if curr in target_pos_set:
            reached_targets.add(curr)
        
        # Optimization: if we found all targets, stop
        if len(reached_targets) == len(target_pos_set):
            break

        for neighbor in get_neighbors(curr, (h, w)):
            if neighbor not in visited:
                tile_id = grid[neighbor]
                # Passable tiles: Empty, Start, Goal, Key.
                # Impassable: Wall, Trap.
                if tile_id != TileType.WALL and tile_id != TileType.TRAP:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return reached_targets

def validate_map(grid):
    """
    Validates map solvability.
    
    Requirements:
    1. Start exists.
    2. Goal exists.
    3. Exactly 3 Keys exist.
    4. All 3 Keys are reachable from Start.
    5. Goal is reachable from AT LEAST ONE Key (implies fully connected path if keys reachable).
       *Strict check*: Ideally Goal is reachable from the "last" collected key, 
       but since any key can be last, Goal must be reachable from the "main connected component" 
       of keys & start. 
       
       So we effectively check:
       - Reachability Start -> {All Keys}
       - Reachability {Any Key} -> Goal
    """
    
    # 1. Locate entities
    start_pos = tuple(map(int, np.argwhere(grid == TileType.START)[0]))
    goal_pos = tuple(map(int, np.argwhere(grid == TileType.GOAL)[0]))
    key_positions = [tuple(map(int, p)) for p in np.argwhere(grid == TileType.KEY)]
    
    if len(key_positions) == 0:
        return False, "No keys found"

    # 2. Reachability: Start -> All Keys
    reached_keys = is_reachable(grid, start_pos, set(key_positions))
    if len(reached_keys) != len(key_positions):
        return False, f"Not all keys reachable from Start ({len(reached_keys)}/{len(key_positions)})"
        
    # 3. Reachability: Any reachable key -> Goal
    # Since we verified all keys are reachable from Start, they are in the same component.
    # We just need to check if Goal is reachable from Start (or any key).
    # NOTE: Goal behaves like a Wall if locked, but for *map validation* (path existence),
    # we assume we can step ONTO the goal once unlocked.
    # The BFS above treats GOAL as passable.
    
    reached_goal = is_reachable(grid, start_pos, {goal_pos})
    if not reached_goal:
        return False, "Goal not reachable from Start"
        
    return True, "Solvable"
