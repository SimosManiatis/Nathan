# Environment Specification

## Tile Types
- **Empty**: Passable space.
- **Wall**: Impassable barrier.
- **Trap**: Hazardous tile.
- **Key**: Collectible item (3 per map).
- **Start**: Agent starting position.
- **Locked Goal**: Exit tile, accessible only after collecting all 3 keys.

## Termination Conditions
- **Success**: Agent reaches the Locked Goal after collecting all 3 keys.
- **Failure (Trap)**: Agent steps onto a Trap tile.
    - *Policy*: Stepping on a trap is terminal (episode ends).
- **Failure (Timeout)**: Agent exceeds the maximum step limit.

## Mechanics
- **Key Collection**: Keys are consumable. Once picked up, the tile becomes Empty.
- **Goal Lock**: The goal tile behaves like a Wall until 3 keys are collected.

## Reward Scheme
- **Step Penalty**: -0.01 per step (encourages efficiency).
- **Key Collection**: +0.5 per key.
- **Goal Reached**: +1.0 (Success).
- **Trap**: -1.0 (Episode Failure).
- **Invalid Move**: 0.0 (Only step penalty applies).

## Invalid Move Policy
- **Wall/Out of Bounds**: Agent stays in the current cell.
- **Locked Goal**: Agent stays in the current cell.

## Episode Limits
- **Max Steps**: `4 * (Width * Height)` (e.g., 256 steps for an 8x8 grid).

