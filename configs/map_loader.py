from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MAPS_DIR = Path(__file__).resolve().parent / "maps"


def _parse_custom_map(rows: list[str], expected_w: int) -> np.ndarray:
    h = len(rows)
    grid = np.zeros((h, expected_w), dtype=np.int8)
    for i, row in enumerate(rows):
        if len(row) != expected_w:
            raise ValueError(
                f"custom_map row {i} length {len(row)} != expected width {expected_w}"
            )
        for j, ch in enumerate(row):
            v = ord(ch) - 48
            if v not in (0, 1, 2):
                raise ValueError(
                    f"custom_map[{i}][{j}]: invalid character '{ch}', "
                    f"expected '0', '1', or '2'"
                )
            grid[i, j] = v
    return grid


def load_map_config(map_id: int) -> dict[str, Any]:
    path = MAPS_DIR / f"map_{map_id}.json"
    if not path.exists():
        raise ValueError(
            f"Map config not found: {path} "
            f"(expected file: configs/maps/map_{map_id}.json)"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sv = data.get("schema_version")
    if sv != 1:
        raise ValueError(
            f"Map {map_id}: unsupported schema_version={sv}, expected 1"
        )

    mid = data.get("map_id")
    if mid != map_id:
        raise ValueError(
            f"Map ID mismatch: file name map_{map_id}.json "
            f"has map_id={mid} in content"
        )

    size = data.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError(
            f"Map {map_id}: size must be a 2-element list, got {size}"
        )

    custom_map = data.get("custom_map")
    if not isinstance(custom_map, list) or len(custom_map) == 0:
        raise ValueError(
            f"Map {map_id}: custom_map must be a non-empty array"
        )

    expected_h, expected_w = size
    if len(custom_map) != expected_h:
        raise ValueError(
            f"Map {map_id}: custom_map row count {len(custom_map)} "
            f"!= size[0] {expected_h}"
        )

    cfg = dict(data)
    cfg["size"] = tuple(size)
    cfg["custom_map"] = _parse_custom_map(custom_map, expected_w)
    return cfg


def load_map_configs(map_ids: list[int]) -> list[dict[str, Any]]:
    return [load_map_config(mid) for mid in map_ids]
