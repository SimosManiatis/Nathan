import pytest
import numpy as np
from gridlock_rl.envs.grid_env import GridEnv
from gridlock_rl.core.constants import TileType, Action

# Helper to create a simple test grid
def create_test_grid(width=5, height=5):
    grid = np.full((height, width), TileType.EMPTY, dtype=np.int8)
    # Start at (0,0)
    grid[0, 0] = TileType.START
    # Key at (0, 2)
    grid[0, 2] = TileType.KEY
    # Trap at (1, 0)
    grid[1, 0] = TileType.TRAP
    # Goal at (4, 4)
    grid[4, 4] = TileType.GOAL
    # Wall at (0, 1)
    grid[0, 1] = TileType.WALL
    return grid

@pytest.fixture
def env():
    return GridEnv(width=5, height=5)

def test_reset_deterministic(env):
    grid = create_test_grid()
    obs, info = env.reset(options={"grid": grid})
    
    # Check agent pos
    assert env.agent_pos == (0, 0)
    # Check grid loaded in obs
    # Agent at (0,0)
    assert obs["grid"][0, 0, 0] == 1 
    # Wall at (0,1)
    assert obs["grid"][1, 0, 1] == 1

def test_movement_and_collision(env):
    grid = create_test_grid()
    env.reset(options={"grid": grid})
    
    # Try move right into Wall (0, 1) -> Should stay at (0,0)
    obs, reward, term, trunc, info = env.step(Action.RIGHT)
    assert env.agent_pos == (0, 0)
    assert info["event"] == "no_op"
    
    # Move Down (0,0) -> (1,0) is Trap
    # Reset first to clear step count etc
    env.reset(options={"grid": grid})
    obs, reward, term, trunc, info = env.step(Action.DOWN)
    
    assert term is True
    assert reward == -1.0
    assert info["event"] == "trap"

def test_key_collection_and_goal_unlock(env):
    # Custom grid: Start(0,0), Key(0,1), Key(0,2), Key(0,3), Goal(0,4)
    grid = np.full((1, 5), TileType.EMPTY, dtype=np.int8)
    grid[0, 0] = TileType.START
    grid[0, 1] = TileType.KEY
    grid[0, 2] = TileType.KEY
    grid[0, 3] = TileType.KEY
    grid[0, 4] = TileType.GOAL
    
    env = GridEnv(width=5, height=1)
    env.reset(options={"grid": grid})
    
    # 1. Collect Key 1
    obs, reward, term, trunc, info = env.step(Action.RIGHT)
    assert reward == 0.5
    assert info["keys_collected"] == 1
    assert obs["keys_collected"][0] == 1
    # Check key removed from grid channel 3
    assert obs["grid"][3, 0, 1] == 0
    
    # 2. Collect Key 2
    env.step(Action.RIGHT)
    assert env.keys_collected == 2
    
    # 3. Collect Key 3
    env.step(Action.RIGHT)
    assert env.keys_collected == 3
    
    # 4. Enter Goal (Now Unlocked)
    obs, reward, term, trunc, info = env.step(Action.RIGHT)
    assert term is True
    assert reward == 1.0
    assert info["event"] == "success"

def test_locked_goal(env):
    # Start(0,0), Goal(0,1). No keys.
    grid = np.full((1, 2), TileType.EMPTY, dtype=np.int8)
    grid[0, 0] = TileType.START
    grid[0, 1] = TileType.GOAL
    
    env = GridEnv(width=2, height=1)
    env.reset(options={"grid": grid})
    
    # Try enter goal
    obs, reward, term, trunc, info = env.step(Action.RIGHT)
    assert env.agent_pos == (0, 0) # Blocked
    assert term is False
    assert info["event"] == "goal_locked"

def test_timeout(env):
    # Max steps = 4 * 2 * 1 = 8
    grid = np.full((1, 2), TileType.EMPTY, dtype=np.int8)
    grid[0, 0] = TileType.START
    
    env = GridEnv(width=2, height=1)
    env.reset(options={"grid": grid})
    
    # Step 8 times
    for _ in range(8):
        obs, r, term, trunc, info = env.step(Action.LEFT) # Wall bonk
        
    assert trunc is True
    assert info["event"] == "timeout"
