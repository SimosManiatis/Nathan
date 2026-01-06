# Observation and Action Spaces

## Action Space
- **Type**: `gymnasium.spaces.Discrete(4)`
- **Actions**:
    - 0: **Up**
    - 1: **Right**
    - 2: **Down**
    - 3: **Left**

## Observation Space
- **Type**: `gymnasium.spaces.Dict`
- **Keys**:
    - `grid`: `Box(0, 1, shape=(Channels, Height, Width), dtype=int8)`
        - **Channels**:
            1. **Agent**: 1 where agent is.
            2. **Walls**: 1 where wall is.
            3. **Traps**: 1 where trap is.
            4. **Keys**: 1 where key is.
            5. **Goal**: 1 where goal is.
    - `keys_collected`: `Box(0, 3, shape=(1,), dtype=int8)`
        - Number of keys currently held by the agent.
