from __future__ import annotations

import numpy as np

MAP_ID = 3

def build_rooms_map(size: int = 128) -> np.ndarray:
    """Multi-room map with doorways connecting rooms.

    4 rooms in a 2x2 grid, separated by walls with doorway gaps.
    Interior is dirty (2), walls are blocked (0).
    """
    grid = np.full((size, size), 2, dtype=np.int8)

    boundary = 1
    grid[0:boundary, :] = 0
    grid[-boundary:, :] = 0
    grid[:, 0:boundary] = 0
    grid[:, -boundary:] = 0

    mid = size // 2
    wall_thickness = 2

    h_start = mid - wall_thickness // 2
    v_start = mid - wall_thickness // 2

    for r in range(h_start, h_start + wall_thickness):
        for c in range(boundary, size - boundary):
            grid[r, c] = 0

    for c in range(v_start, v_start + wall_thickness):
        for r in range(boundary, size - boundary):
            grid[r, c] = 0

    door_size = 3
    half_door = door_size // 2

    # Horizontal wall doors (left and right)
    for center_c in [mid // 2, mid + mid // 2]:
        for r in range(h_start, h_start + wall_thickness):
            for c in range(center_c - half_door, center_c + half_door + 1):
                if not (0 <= r < size and 0 <= c < size):
                    continue
                if v_start <= c < v_start + wall_thickness:
                    continue
                grid[r, c] = 2

    # Vertical wall doors (top and bottom)
    for center_r in [mid // 2, mid + mid // 2]:
        for c in range(v_start, v_start + wall_thickness):
            for r in range(center_r - half_door, center_r + half_door + 1):
                if not (0 <= r < size and 0 <= c < size):
                    continue
                if h_start <= r < h_start + wall_thickness:
                    continue
                grid[r, c] = 2

    return grid

MAP_CONFIG = {
    "size": (128, 128),
    "custom_map": build_rooms_map(128),
    "agent_spawn_pool": [
        (2, 2),
        (2, 125),
        (125, 2),
        (125, 125),
    ],
    "npc_spawn_pool": [
        (32, 32),
        (96, 32),
        (32, 96),
        (96, 96),
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
