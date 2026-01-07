import pytest
import numpy as np
from gridlock_rl.envs.grid_env import GridEnv
from gridlock_rl.core.constants import TileType, Action

def create_1key_map(width=5, height=1):
    """
    S . K . G
    0 1 2 3 4
    """
    grid = np.full((height, width), TileType.EMPTY, dtype=np.int8)
    grid[0, 0] = TileType.START
    grid[0, 2] = TileType.KEY
    grid[0, 4] = TileType.GOAL
    return grid

def test_goal_unlock_uses_total_keys():
    """
    Goal unlock uses total_keys
    For a 1-key map: after collecting the single key, stepping onto goal must terminate with success.
    """
    grid = create_1key_map()
    # Explicitly set total_keys=1 via env config or verify it detects it
    # The current env might default to 3 keys, so we need to ensure it adapts to the map
    # OR we pass the correct total_keys in the constructor if that's how it's designed.
    # Looking at grid_env.py, it likely counts keys on reset OR has a fixed param.
    # Initial scan showed `self.total_keys = 3` hardcoded. This test SHOULD fail if that's true.
    
    env = GridEnv(width=5, height=1, total_keys=1) # Passing arg in case we fix it to accept it
    env.reset(options={"grid": grid})
    
    # 1. Move to Key (Right, Right)
    env.step(Action.RIGHT)
    obs, reward, term, trunc, info = env.step(Action.RIGHT)
    
    assert info["keys_collected"] == 1
    assert "key_collected" in info["event"]
    
    # 2. Check if Goal is unlocked
    # Move Right to (0,3)
    env.step(Action.RIGHT)
    # Move Right to (0,4) -> Goal
    obs, reward, term, trunc, info = env.step(Action.RIGHT)
    
    assert term is True, "Episode should terminate on entering unlocked goal"
    assert info["event"] == "success", "Event should be success"
    assert reward > 0, "Reward should be positive (success reward)"

def test_dense_shaping_sanity():
    """
    Dense shaping sanity
    When the agent takes a step that strictly reduces the BFS distance to the current target, 
    the shaping reward must be > 0 (and the opposite step < 0), except on target switch events.
    """
    width = 5
    height = 1
    grid = np.full((height, width), TileType.EMPTY, dtype=np.int8)
    # S . . K G
    grid[0, 0] = TileType.START
    grid[0, 3] = TileType.KEY
    grid[0, 4] = TileType.GOAL
    
    env = GridEnv(width=width, height=height, dense_reward=True, total_keys=1) # Ensure dense reward ON
    env.reset(options={"grid": grid})
    
    # Initial State: Agent at (0,0). Target Key at (0,3). Dist = 3.
    # Move Right -> (0,1). New Dist = 2. Shaping should be positive.
    
    obs, reward, term, trunc, info = env.step(Action.RIGHT)
    # Total Reward = Step_Cost + Shaping
    # We want to isolate Shaping. 
    # But usually Shaping > Step_Cost. 
    # Let's check the logic:
    # If dense_reward is ON, reward = -step_cost + (new_pot - old_pot)
    # new_pot should be > old_pot.
    
    # Check if reward is strictly greater than -step_cost (implying positive shaping)
    # Or simpler: check if reward > 0 (assuming shaping dominates step cost, which is typical)
    # But strictly: (new_pot - old_pot) > 0
    
    # We can't access potentials directly unless we access private methods or inferred from reward.
    # Let's inspect internal state for verification
    prev_pot = env.last_potential # This is AFTER the step, so we can't see "prev" easily without hacking.
    # Actually, env.last_potential is updated at the *end* of step.
    
    # Let's just assert the reward logic.
    step_cost = env.step_cost
    shaping = reward + step_cost
    
    assert shaping > 0.0, f"Moving closer to target should yield positive shaping. Got shaping={shaping}"
    
    # Now move Left -> (0,0). Dist increases 2 -> 3. Shaping should be negative.
    obs, reward, term, trunc, info = env.step(Action.LEFT)
    shaping = reward + step_cost
    assert shaping < 0.0, f"Moving away from target should yield negative shaping. Got shaping={shaping}"

