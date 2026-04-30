from __future__ import annotations

MAP_ID = 1

import numpy as np


def build_map(size: int = 128) -> np.ndarray:
    """Map 1: 对称, 充电站几乎可见.

    Layout:
    - 128x128 grid, boundary is blocked (0)
    - Four edge-middle obstacles (near but not touching boundary)
    - No central cross corridor (open center for free movement)
    - Everything else is dirty (2)
    """
    if size <= 2:
        raise ValueError("size must be greater than 2.")

    grid = np.full((size, size), 2, dtype=np.int8)
    # Boundary walls
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0

    # Four corner obstacles (near corners but not touching boundary walls)
    # Top-left corner
    grid[0:32, 0:32] = 0
    # Top-right corner
    grid[0:32, 96:128] = 0
    # Bottom-left corner
    grid[96:128, 0:32] = 0
    # Bottom-right corner
    grid[96:128, 96:128] = 0
    # Center
    grid[56:72, 56:72] = 0

    return grid


MAP_CONFIG = {
    "size": (128, 128),
    "custom_map": build_map(128),

    # Agent spawn points: edge only (for charger-memory validation)
    # Center spawn removed — agent must see charger from edge spawn
    "agent_spawn_pool": [
        (64, 1),    # Top edge
        (64, 126),  # Bottom edge
        (1, 64),    # Left edge
        (126, 64),  # Right edge
    ],

    # NPC spawn points: 4 corners, separated enough so walk radii don't overlap
    # Each NPC walks in a 21x21 area (radius=10). Distance between any two > 20.
    "npc_spawn_pool": [
        (33, 33),    # Top-left
        (95, 33),   # Top-right
        (33, 95),   # Bottom-left
        (95, 95),  # Bottom-right
    ],

    # Charging stations: placed within edge spawn's field of view (radius 10)
    # So agent can see charger immediately after birth
    "station_pool": [
        {"x": 62, "z": 8, "dx": 3, "dz": 3},    # Top edge spawn (64,1) can see
        {"x": 62, "z": 118, "dx": 3, "dz": 3},  # Bottom edge spawn (64,126) can see
        {"x": 8, "z": 62, "dx": 3, "dz": 3},    # Left edge spawn (1,64) can see
        {"x": 118, "z": 62, "dx": 3, "dz": 3},  # Right edge spawn (126,64) can see
    ],

    "max_battery": 200,
    "max_steps": 1000,
    "hero_id": 37,
    "npc_ids": None,
    "map_id": 1,
    "local_view_size": 21,
}
