"""从 Python 地图源文件导出为运行时 JSON。

源文件目录: configs/maps/src/map_{id}.py  (不参与运行时加载)
输出目录:   configs/maps/map_{id}.json    (运行时唯一数据来源)

修改地图后请按顺序执行:
  1. 编辑 configs/maps/src/map_{id}.py
  2. 运行此脚本  →  更新 configs/maps/map_{id}.json
  3. 运行 validate_maps_json.py 确认无错
  4. 开始训练
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MAP_IDS = [1, 2, 3, 4]
OUTPUT_DIR = PROJECT_ROOT / "configs" / "maps"
SCHEMA_VERSION = 1


def _load_map_config_from_source(map_id: int) -> dict:
    module_name = f"configs.maps.src.map_{map_id}"
    mod = importlib.import_module(module_name)
    cfg = dict(getattr(mod, "MAP_CONFIG"))
    expected_map_id = getattr(mod, "MAP_ID", None)
    if expected_map_id is not None and expected_map_id != map_id:
        raise ValueError(
            f"Map ID mismatch: file map_{map_id}.py has MAP_ID={expected_map_id}"
        )
    cfg["map_id"] = map_id
    return cfg


def _grid_to_strings(grid: np.ndarray) -> list[str]:
    return ["".join(str(int(v)) for v in row) for row in grid]


def _convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def export_map(map_id: int) -> dict:
    cfg = _load_map_config_from_source(map_id)
    grid = cfg["custom_map"]

    agent_pool = [_convert(p) for p in cfg.get("agent_spawn_pool", [])]
    npc_pool = [_convert(p) for p in cfg.get("npc_spawn_pool", [])]
    station_pool = cfg.get("station_pool", [])

    return {
        "schema_version": SCHEMA_VERSION,
        "map_id": map_id,
        "size": list(cfg["size"]),
        "custom_map": _grid_to_strings(grid),
        "agent_spawn_pool": agent_pool,
        "npc_spawn_pool": npc_pool,
        "station_pool": station_pool,
        "max_battery": int(cfg.get("max_battery", 200)),
        "max_steps": int(cfg.get("max_steps", 1000)),
        "hero_id": int(cfg.get("hero_id", 37)),
        "npc_ids": cfg.get("npc_ids"),
        "local_view_size": int(cfg.get("local_view_size", 21)),
    }


class _CompactEncoder(json.JSONEncoder):
    def encode(self, o):
        if isinstance(o, dict):
            return self._encode_dict(o, 0)
        return super().encode(o)

    def _encode_dict(self, d: dict, depth: int) -> str:
        indent = "  "
        items = []
        for k, v in d.items():
            key_str = json.dumps(k, ensure_ascii=False)
            if k == "custom_map":
                val_str = self._encode_custom_map(v, depth + 1)
            elif isinstance(v, list) and v and all(isinstance(x, list) for x in v):
                rows = []
                for row in v:
                    rows.append(indent * (depth + 2) + json.dumps(row))
                val_str = "[\n" + ",\n".join(rows) + "\n" + indent * (depth + 1) + "]"
            elif isinstance(v, dict):
                val_str = self._encode_dict(v, depth + 1)
            elif isinstance(v, str):
                val_str = json.dumps(v, ensure_ascii=False)
            elif isinstance(v, list):
                val_str = json.dumps(v, ensure_ascii=False)
            else:
                val_str = json.dumps(v)
            items.append(indent * (depth + 1) + key_str + ": " + val_str)
        return "{\n" + ",\n".join(items) + "\n" + indent * depth + "}"

    def _encode_custom_map(self, rows: list[str], depth: int) -> str:
        indent = "  "
        lines = ",\n".join(indent * (depth + 1) + json.dumps(r) for r in rows)
        return "[\n" + lines + "\n" + indent * depth + "]"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for map_id in MAP_IDS:
        data = export_map(map_id)
        out_path = OUTPUT_DIR / f"map_{map_id}.json"

        encoder = _CompactEncoder()
        content = encoder.encode(data)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
            f.write("\n")

        h = len(data["custom_map"])
        w = len(data["custom_map"][0]) if h > 0 else 0
        print(f"  [OK] map_{map_id}.json  size=({w}, {h})  "
              f"agents={len(data['agent_spawn_pool'])}  "
              f"npcs={len(data['npc_spawn_pool'])}  "
              f"stations={len(data['station_pool'])}")

    print(f"\nAll maps exported to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
