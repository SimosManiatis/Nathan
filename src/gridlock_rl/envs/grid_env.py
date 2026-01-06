import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium import spaces

from gridlock_rl.core.constants import TileType, Action, CHANNEL_MAP
from gridlock_rl.maps.generator import MapGenerator

class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "ascii"], "render_fps": 4}

    def __init__(self, render_mode=None, width=8, height=8, trap_density=0.1, 
                 max_width=None, max_height=None, dense_reward=False, 
                 success_reward=20.0, key_reward=2.0, trap_cost=20.0, step_cost=0.01, timeout_penalty=10.0,
                 max_steps_multiplier=4):
        super().__init__()
        self.width = width
        self.height = height
        self.max_width = max_width or width
        self.max_height = max_height or height
        self.render_mode = render_mode
        self.use_dense_reward = dense_reward
        
        # Reward coefficients
        self.success_reward = success_reward
        self.key_reward = key_reward
        self.trap_cost = trap_cost
        self.step_cost = step_cost
        self.total_keys = 3  # Phase 7 Fix: default 3 keys
        self.timeout_penalty = timeout_penalty
        
        self.max_steps_multiplier = max_steps_multiplier
        
        self.map_generator = MapGenerator(width=width, height=height, trap_density=trap_density)

        # Action Space: 4 discrete actions (Up, Right, Down, Left)
        self.action_space = spaces.Discrete(len(Action))

        # Observation Space: Dict with 'grid' and 'keys_collected'
        # Grid: C x H x W (Channels: Agent, Wall, Trap, Key, Goal)
        n_channels = len(CHANNEL_MAP)
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0, high=1, 
                shape=(n_channels, self.max_height, self.max_width), 
                dtype=np.int8
            ),
            "keys_collected": spaces.Box(
                low=0, high=3, 
                shape=(1,), 
                dtype=np.int8
            )
        })

        # Internal State
        self.grid_static = None  # Reference to initial layout
        self.grid_dynamic = None # Current state of the world (keys removed)
        self.agent_pos = None    # (row, col)
        self.keys_collected = 0
        self.steps = 0
        self.steps = 0
        self.last_potential = 0.0
        self.max_steps = self.max_steps_multiplier * (width * height)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Seeds self.np_random
        
        # 1. Map Generation / Loading
        if options and "grid" in options:
            # Deterministic reset for testing
            self.grid_static = np.array(options["grid"], dtype=np.int8)
            # Validation? Maybe later. Assume valid for now.
        else:
            # Generate new map using env's RNG seed logic if needed
            # MapGenerator uses global np.random or specific seed.
            # We can pass a seed derived from self.np_random
            gen_seed = int(self.np_random.integers(0, 2**32))
            self.grid_static, _ = self.map_generator.generate(seed=gen_seed)

        # 2. State Initialization
        self.grid_dynamic = self.grid_static.copy()
        
        # Locate agent
        start_indices = np.argwhere(self.grid_static == TileType.START)
        if len(start_indices) == 0:
            raise ValueError("Map missing START tile")
        self.agent_pos = tuple(start_indices[0])
        
        self.keys_collected = 0
        self.keys_collected = 0
        self.steps = 0
        self.last_potential = self._compute_potential() if self.use_dense_reward else 0.0
        
        return self._get_obs(), self._get_info(event="reset")

    def step(self, action):
        self.steps += 1
        reward = -self.step_cost # Step penalty
        terminated = False
        truncated = False
        event = "moved"

        # 1. Calculate new position
        r, c = self.agent_pos
        dr, dc = 0, 0
        if action == Action.UP: dr = -1
        elif action == Action.DOWN: dr = 1
        elif action == Action.LEFT: dc = -1
        elif action == Action.RIGHT: dc = 1
        
        nr, nc = r + dr, c + dc
        
        # 2. Validation (Bounds, Walls, Locked Goal)
        next_tile = TileType.EMPTY # Default if out of bounds (treated as wall below)
        valid_move = True
        
        if not (0 <= nr < self.height and 0 <= nc < self.width):
            valid_move = False # Out of bounds
            event = "no_op"
        else:
            next_tile = self.grid_dynamic[nr, nc]
            if next_tile == TileType.WALL:
                valid_move = False
                event = "no_op"
            elif next_tile == TileType.GOAL:
                if self.keys_collected < self.total_keys:
                    valid_move = False
                    event = "goal_locked"
        
        if valid_move:
            self.agent_pos = (nr, nc)
            
            # 3. Interactions
            if next_tile == TileType.TRAP:
                reward = -self.trap_cost
                terminated = True
                event = "trap"
            
            elif next_tile == TileType.KEY:
                reward = self.key_reward
                self.keys_collected += 1
                # Remove key from dynamic grid
                self.grid_dynamic[nr, nc] = TileType.EMPTY
                event = "key_collected"
            
            elif next_tile == TileType.GOAL:
                # Can only enter if unlocked (checked above)
                reward = self.success_reward
                terminated = True
                event = "success"

        # 3.5 Dense Reward Shaping
        if self.use_dense_reward and not terminated:
            current_potential = self._compute_potential()
            shaping = current_potential - self.last_potential
            reward += shaping
            self.last_potential = current_potential

        # 4. Truncation
        if self.steps >= self.max_steps:
            truncated = True
            if not terminated:
                reward -= self.timeout_penalty
                event = "timeout"

        # 5. Render
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info(event)

    def _get_obs(self):
        # Create multi-channel grid with PADDED size
        obs_grid = np.zeros((len(CHANNEL_MAP), self.max_height, self.max_width), dtype=np.int8)
        
        # Fill channels based on dynamic grid
        # 0: Agent
        ar, ac = self.agent_pos
        obs_grid[CHANNEL_MAP["agent"], ar, ac] = 1
        
        # 1: Wall
        obs_grid[CHANNEL_MAP["wall"], :self.height, :self.width] = (self.grid_dynamic == TileType.WALL).astype(np.int8)
        
        # 2: Trap
        obs_grid[CHANNEL_MAP["trap"], :self.height, :self.width] = (self.grid_dynamic == TileType.TRAP).astype(np.int8)
        
        # 3: Key
        obs_grid[CHANNEL_MAP["key"], :self.height, :self.width] = (self.grid_dynamic == TileType.KEY).astype(np.int8)
        
        # 4: Goal
        obs_grid[CHANNEL_MAP["goal"], :self.height, :self.width] = (self.grid_dynamic == TileType.GOAL).astype(np.int8)
        
        return {
            "grid": obs_grid,
            "keys_collected": np.array([self.keys_collected], dtype=np.int8)
        }

    def _get_info(self, event):
        return {
            "event": event,
            "keys_collected": self.keys_collected,
            "steps": self.steps
        }

    def _compute_potential(self):
        # Potential-based shaping: Phi(s) = -Distance(agent, nearest_target)
        # Targets: Keys (if remaining) or Goal (if all keys collected)
        
        target_indices = []
        if self.keys_collected < self.total_keys:
            target_indices = np.argwhere(self.grid_dynamic == TileType.KEY)
        else:
            target_indices = np.argwhere(self.grid_dynamic == TileType.GOAL)
            
        if len(target_indices) == 0:
            return 0.0
            
        targets = [tuple(t) for t in target_indices]
        
        # Simple BFS to find shortest path to ANY target
        # Treating Walls as blocking, Traps as blocking (safe path)
        queue = deque([(self.agent_pos, 0)])
        visited = {self.agent_pos}
        
        min_dist = float('inf')
        
        while queue:
            curr, dist = queue.popleft()
            
            if curr in targets:
                min_dist = dist
                break
                
            r, c = curr
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in visited:
                        tile = self.grid_dynamic[nr, nc]
                        # Passable: Empty, Start, Goal (if unlocked?), Key
                        # Impassable: Wall. Trap? Let's say Trap is impassable for "Safe Distance"
                        if tile != TileType.WALL and tile != TileType.TRAP:
                             # Also consider Goal impassable if locked?
                             # Actually logic: if searching for keys, Goal is obstacle? 
                             # Simplification: Treat Goal as empty unless it is target.
                             # But wait, step() logic blocks locked goal.
                             if tile == TileType.GOAL and self.keys_collected < self.total_keys:
                                 pass # treat as wall
                             else:
                                visited.add((nr, nc))
                                queue.append(((nr, nc), dist + 1))
                                
        if min_dist == float('inf'):
            # No path found (isolated?), return worst case
            # Max possible dist is W*H
            min_dist = self.width * self.height
            
        # Normalize potential to be roughly in range [0, 1] or similar scale
        # Phi(s) should be higher when closer.
        # Let's say Phi = 1.0 - (dist / max_dist)
        norm_dist = min_dist / (self.width * self.height)
        return -norm_dist # Negative distance is potential (closer = less negative = higher reward)

    def render(self):
        if self.render_mode is None:
            return

        # Simple ASCII Render
        output = []
        output.append("-" * (self.width + 2))
        for r in range(self.height):
            line = "|"
            for c in range(self.width):
                if (r, c) == self.agent_pos:
                    line += "A"
                else:
                    tile = self.grid_dynamic[r, c]
                    if tile == TileType.EMPTY: line += " "
                    elif tile == TileType.WALL: line += "#"
                    elif tile == TileType.START: line += "S" # Usually empty after start
                    elif tile == TileType.GOAL: line += "G"
                    elif tile == TileType.KEY: line += "K"
                    elif tile == TileType.TRAP: line += "x"
            line += "|"
            output.append(line)
        output.append("-" * (self.width + 2))
        
        text = "\n".join(output)
        if self.render_mode == "human":
            print(text)
        return text
