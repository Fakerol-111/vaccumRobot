from __future__ import annotations

import numpy as np

MAP_ID = 2

def build_maze_map(size: int = 128) -> np.ndarray:
    """Maze-like map with corridor grid pattern.

    A grid of obstacle blocks with narrow corridors between them.
    Interior paths are dirty (2), walls are blocked (0).
    """
    grid = np.full((size, size), 2, dtype=np.int8)

    boundary = 1
    grid[0:boundary, :] = 0
    grid[-boundary:, :] = 0
    grid[:, 0:boundary] = 0
    grid[:, -boundary:] = 0

    block_size = 12
    corridor_width = 3
    start = boundary + 4

    for row_start in range(start, size - boundary, block_size + corridor_width):
        for col_start in range(start, size - boundary, block_size + corridor_width):
            r_end = min(row_start + block_size, size - boundary)
            c_end = min(col_start + block_size, size - boundary)
            for r in range(row_start, r_end):
                for c in range(col_start, c_end):
                    if 0 <= r < size and 0 <= c < size:
                        grid[r, c] = 0

    for r in range(boundary + 2, size - boundary - 1, block_size + corridor_width):
        corridor_center = r + block_size
        for c in range(boundary, size - boundary):
            if 0 <= corridor_center < size and 0 <= c < size:
                grid[corridor_center, c] = 2

    for c in range(boundary + 2, size - boundary - 1, block_size + corridor_width):
        corridor_center = c + block_size
        for r in range(boundary, size - boundary):
            if 0 <= r < size and 0 <= corridor_center < size:
                grid[r, corridor_center] = 2

    return grid

MAP_CONFIG = {
    "size": (128, 128),
    "custom_map": build_maze_map(128),
    "agent_spawn_pool": [
        (2, 2),
        (2, 125),
        (125, 2),
        (62, 62),
        (62, 77),
        (77, 62),
    ],
    "npc_spawn_pool": [
        (18, 64),
        (34, 64),
        (49, 64),
        (64, 34),
        (64, 49),
        (64, 94),
        (79, 64),
        (94, 64),
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
