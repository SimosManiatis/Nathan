import numpy as np
import random
from gridlock_rl.core.constants import TileType
from gridlock_rl.maps.validation import validate_map

class MapGenerator:
    def __init__(self, width=8, height=8, trap_density=0.1, max_retries=100, num_keys=3, min_traps=0):
        self.width = width
        self.height = height
        self.trap_density = trap_density
        self.max_retries = max_retries
        self.num_keys = num_keys
        self.min_traps = min_traps

    def generate(self, seed=None):
        """
        Generates a valid map.
        Returns:
            grid (np.ndarray): The generated grid.
            info (dict): Metadata including seed and retry count.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        for attempt in range(self.max_retries):
            grid = np.full((self.height, self.width), TileType.EMPTY, dtype=np.int8)
            
            # Helper to get empty positions
            def get_empty_coords():
                return [tuple(x) for x in np.argwhere(grid == TileType.EMPTY)]

            # 1. Place Wall Borders (Optional, but good for gridworlds)
            # For this simple version, let's keep it open or just rely on bounds logic.
            # Let's add random inner walls? Or just traps? 
            # Request says "traps", "walls if any". Let's stick to Traps for difficulty.

            all_coords = [(r, c) for r in range(self.height) for c in range(self.width)]
            random.shuffle(all_coords)
            
            # 2. Place Unique Items
            # Needs: 1 Start, 1 Goal, N Keys
            if len(all_coords) < 2 + self.num_keys:
                raise ValueError("Grid too small")
                
            start_pos = all_coords.pop()
            goal_pos = all_coords.pop()
            key_positions = [all_coords.pop() for _ in range(self.num_keys)]
            
            grid[start_pos] = TileType.START
            grid[goal_pos] = TileType.GOAL
            for kp in key_positions:
                grid[kp] = TileType.KEY
                
            # 3. Place Traps
            # Remaining empty cells
            remaining = all_coords # already shuffled and popped
            n_traps_density = int(len(remaining) * self.trap_density)
            n_traps = max(self.min_traps, n_traps_density)
            
            for _ in range(n_traps):
                if not remaining: break
                trap_pos = remaining.pop()
                grid[trap_pos] = TileType.TRAP
                
            # 4. Validate
            is_valid, msg = validate_map(grid)
            if is_valid:
                return grid, {"seed": seed, "attempts": attempt + 1}
                
        raise RuntimeError(f"Failed to generate solvable map after {self.max_retries} attempts")

if __name__ == "__main__":
    # Quick standalone test
    gen = MapGenerator(width=8, height=8, trap_density=0.2)
    try:
        grid, info = gen.generate(seed=42)
        print("Generated Map:")
        print(grid)
        print(info)
        print("\nLegend: 0=Empty, 1=Wall, 2=Start, 3=Goal, 4=Key, 5=Trap")
    except Exception as e:
        print(e)
