from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent


def load_map_config(map_id: int) -> dict[str, Any]:
    module_name = f"config.map_{map_id}"
    try:
        mod = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Map config not found: {module_name} (expected file: config/map_{map_id}.py)")

    cfg = dict(getattr(mod, "MAP_CONFIG"))
    expected_map_id = getattr(mod, "MAP_ID", None)
    if expected_map_id is not None and expected_map_id != map_id:
        raise ValueError(
            f"Map ID mismatch: file map_{map_id}.py has MAP_ID={expected_map_id}"
        )
    cfg["map_id"] = map_id
    return cfg


def load_map_configs(map_ids: list[int]) -> list[dict[str, Any]]:
    configs = []
    for mid in map_ids:
        configs.append(load_map_config(mid))
    return configs
