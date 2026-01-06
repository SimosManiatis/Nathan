from enum import IntEnum

class TileType(IntEnum):
    EMPTY = 0
    WALL = 1
    START = 2
    GOAL = 3
    KEY = 4
    TRAP = 5

class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# Channel mapping for observation tensor
CHANNEL_MAP = {
    "agent": 0,
    "wall": 1,
    "trap": 2,
    "key": 3,
    "goal": 4
}
