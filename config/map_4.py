from __future__ import annotations

import numpy as np

MAP_ID = 4

def build_open_scattered_map(size: int = 128, seed: int = 42) -> np.ndarray:
    """Open map with scattered random obstacles, no boundary walls.

    Interior is mostly dirty (2) with scattered blocked clusters (0).
    """
    rng = np.random.RandomState(seed)
    grid = np.full((size, size), 2, dtype=np.int8)

    num_clusters = 18
    for _ in range(num_clusters):
        cx = rng.randint(4, size - 4)
        cy = rng.randint(4, size - 4)
        cluster_w = rng.randint(2, 5)
        cluster_h = rng.randint(2, 5)
        for i in range(cluster_h):
            for j in range(cluster_w):
                x = cx + j
                y = cy + i
                if 0 <= x < size and 0 <= y < size:
                    grid[y, x] = 0

    return grid

MAP_CONFIG = {
    "size": (128, 128),
    "custom_map": build_open_scattered_map(128),
    "agent_spawn_pool": [
        (2, 64),
        (125, 64),
        (64, 2),
        (64, 125),
    ],
    "npc_spawn_pool": [
        (10, 10),
        (10, 117),
        (117, 10),
        (117, 117),
        (64, 64),
    ],
    "station_pool": [
        {"x": 2, "z": 2, "dx": 3, "dz": 3},
        {"x": 123, "z": 2, "dx": 3, "dz": 3},
        {"x": 2, "z": 123, "dx": 3, "dz": 3},
        {"x": 123, "z": 123, "dx": 3, "dz": 3},
    ],
    "max_battery": 200,
    "max_steps": 1000,
    "hero_id": 37,
    "npc_ids": None,
    "map_id": MAP_ID,
    "local_view_size": 21,
}
