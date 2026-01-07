import numpy as np
from collections import deque
from gridlock_rl.envs.grid_env import GridEnv
from gridlock_rl.core.constants import TileType, Action

def bfs_path(grid, start, target_type):
    """
    Finds shortest path from start (r, c) to nearest tile of target_type.
    Treats Walls and Traps as impassable.
    Returns list of Actions.
    """
    rows, cols = grid.shape
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        (r, c), path = queue.popleft()
        
        if grid[r, c] == target_type:
            return path
        
        # Explore neighbors
        # Priority: Up, Right, Down, Left (arbitrary)
        moves = [
            (Action.UP, (-1, 0)),
            (Action.RIGHT, (0, 1)),
            (Action.DOWN, (1, 0)),
            (Action.LEFT, (0, -1))
        ]
        
        for action, (dr, dc) in moves:
            nr, nc = r + dr, c + dc
            
            if 0 <= nr < rows and 0 <= nc < cols:
                tile = grid[nr, nc]
                if (nr, nc) not in visited:
                    # Check passable
                    if tile != TileType.WALL and tile != TileType.TRAP:
                        # Note: Goal is passable here? 
                        # If target is GOAL, obviously.
                        # If target is KEY, we can walk through GOAL? 
                        # In Env: Goal is locked if keys not collected.
                        # So if target is KEY, Goal is effectively a Wall.
                        if target_type == TileType.KEY and tile == TileType.GOAL:
                            pass # Treat as wall
                        else:
                            visited.add((nr, nc))
                            new_path = path + [action]
                            queue.append(((nr, nc), new_path))
                            
    return None # No path found

def run_oracle():
    print("Running Oracle Rollout on Stage 0B Map (Traps)...")
    
    # 1. Generate Stage 0B Map using Generator (6x6, 1 Key, 0.05 Traps)
    # This ensures we test the actual generation logic + solvability.
    from gridlock_rl.maps.generator import MapGenerator
    
    gen = MapGenerator(width=6, height=6, trap_density=0.05, num_keys=1)
    # We loop until we get a map with at least one trap to verify behavior
    grid = None
    for i in range(20):
        grid, _ = gen.generate(seed=i)
        if np.any(grid == TileType.TRAP):
            print(f"Found map with traps at seed {i}")
            break
            
    if grid is None:
        print("Could not generate map with traps in 20 tries (expected for low density). Using last.")
    
    env = GridEnv(width=6, height=6, dense_reward=True, num_keys=1)
    obs, info = env.reset(options={"grid": grid})
    
    print("\nInitial State:")
    env.render()
    
    # 2. Plan Path
    start_pos = env.agent_pos
    
    # Leg 1: Start -> Key
    print("\nPlanning Leg 1: Start -> Key")
    path_to_key = bfs_path(grid, start_pos, TileType.KEY)
    if not path_to_key:
        print("ERROR: No path to Key found!")
        return
        
    print(f"Path to Key: {[a.name for a in path_to_key]}")
    
    # Leg 2: Key -> Goal
    # Simulate end position
    curr = start_pos
    for action in path_to_key:
        dr, dc = 0, 0
        if action == Action.UP: dr = -1
        elif action == Action.DOWN: dr = 1
        elif action == Action.LEFT: dc = -1
        elif action == Action.RIGHT: dc = 1
        curr = (curr[0] + dr, curr[1] + dc)
        
    print("\nPlanning Leg 2: Key -> Goal")
    path_to_goal = bfs_path(grid, curr, TileType.GOAL)
    if not path_to_goal:
        print("ERROR: No path to Goal found! (This implies MapGenerator validation failed or BFS mismatch)")
        return
        
    print(f"Path to Goal: {[a.name for a in path_to_goal]}")
    
    full_plan = path_to_key + path_to_goal
    
    # 3. Execute
    total_reward = 0.0
    terminated = False
    truncated = False
    
    print("\nExecuting Plan...")
    for i, action in enumerate(full_plan):
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # print(f"Step {i+1}: {action.name}, Reward: {reward:.4f}, Event: {info['event']}")
        
        if terminated or truncated:
            break
            
    # 4. Assertions
    print("\nResults:")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}")
    print(f"Total Reward: {total_reward:.4f}")
    
    assert terminated, "Oracle failed to finish episode"
    assert info["event"] == "success", f"Oracle failed with event: {info['event']}"
    assert info["keys_collected"] == 1, "Oracle missed the key"
    
    print("\nSUCCESS: Oracle cleaned the Stage 0B map.")

if __name__ == "__main__":
    run_oracle()
