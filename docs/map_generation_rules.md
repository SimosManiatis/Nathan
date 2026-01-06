# Map Generation Rules

## Constraints
- **Placement**: Keys cannot spawn on Traps, Start, or Goal.
- **Solvability**: A valid path MUST exist that:
    1. Starts at Start.
    2. Visits all 3 Key locations.
    3. Ends at Goal.
    4. Does not pass through Walls or Traps.

## Generator Logic (High Level)
1. **Grid Initialization**: Create empty grid with borders.
2. **Object Placement**:
    - Place Start and Goal (guaranteed not to overlap).
    - Place 3 Keys (guaranteed distinct and not on Start/Goal).
    - Place Traps (density parameter).
3. **Validation**:
    - Run BFS/A* to verify reachability of all 3 keys from Start.
    - Verify reachability of Goal from each Key location.
    - If validation fails, regenerate or retry placement.
