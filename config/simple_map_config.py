from __future__ import annotations

import numpy as np


def build_simple_map(size: int = 128) -> np.ndarray:
    """Create a simple square map.

    Boundary cells are blocked (0), and inner cells are dirty (2).
    """
    if size <= 2:
        raise ValueError("size must be greater than 2.")

    grid = np.full((size, size), 2, dtype=np.int8)
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, 0] = 0
    grid[:, -1] = 0
    return grid


SIMPLE_MAP_CONFIG = {
    "size": (128, 128),
    "custom_map": build_simple_map(128),

    "agent_spawn_pool": [
        (64, 1),
        (64, 126),
        (1, 64),
        (126, 64),
    ],

    "npc_spawn_pool": [
        (64, 64),
        (32, 32),
        (96, 96),
        (32, 96),
        (96, 32),
    ],

    "station_pool": [
        {"x": 1, "z": 1, "dx": 3, "dz": 3},
        {"x": 124, "z": 1, "dx": 3, "dz": 3},
        {"x": 1, "z": 124, "dx": 3, "dz": 3},
        {"x": 124, "z": 124, "dx": 3, "dz": 3},
    ],

    "max_battery": 200,
    "max_steps": 1000,
    "hero_id": 37,
    "npc_ids": None,
    "map_id": 1,
    "local_view_size": 21,
}
