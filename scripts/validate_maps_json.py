"""校验 configs/maps/map_{id}.json 文件完整性。

检查内容:
  - 所有 required 字段存在
  - schema_version 为 1
  - custom_map 行列数与 size 匹配
  - 所有坐标不越界
  - 字符合法性（仅 0/1/2）

在 export_maps_to_json.py 之后、训练之前运行。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MAPS_DIR = PROJECT_ROOT / "configs" / "maps"
SCHEMA_FILE = MAPS_DIR / "schema.json"


def _check_dimensions(data: dict) -> list[str]:
    errors = []
    size = data.get("size", [])
    custom_map = data.get("custom_map", [])

    if len(size) != 2:
        errors.append(f"size should have 2 elements, got {len(size)}")

    expected_h, expected_w = size[0] if len(size) == 2 else (0, 0), size[1] if len(size) == 2 else (0, 0)
    actual_h = len(custom_map)
    if actual_h != expected_h:
        errors.append(f"custom_map row count {actual_h} != size[0] {expected_h}")

    for i, row in enumerate(custom_map):
        if not isinstance(row, str):
            errors.append(f"custom_map row {i} is not a string (got {type(row).__name__})")
            continue
        if len(row) != expected_w:
            errors.append(f"custom_map row {i} has {len(row)} cols != expected {expected_w}")
        if not re.match(r"^[012]+$", row):
            invalid = set(row) - {"0", "1", "2"}
            errors.append(f"custom_map row {i} has invalid chars: {sorted(invalid)}")

    return errors


def _check_pools(data: dict) -> list[str]:
    errors = []
    size = data.get("size", [0, 0])
    w, h = size[0] if len(size) == 2 else 0, size[1] if len(size) == 2 else 0

    for name in ("agent_spawn_pool", "npc_spawn_pool"):
        pool = data.get(name, [])
        for i, p in enumerate(pool):
            if len(p) != 2:
                errors.append(f"{name}[{i}] should have 2 elements, got {len(p)}")
            else:
                x, z = p
                if not (0 <= x < w and 0 <= z < h):
                    errors.append(f"{name}[{i}] ({x}, {z}) out of bounds ({w}x{h})")

    for i, s in enumerate(data.get("station_pool", [])):
        x, z = s.get("x", -1), s.get("z", -1)
        dx, dz = s.get("dx", 0), s.get("dz", 0)
        if not (0 <= x < w and 0 <= z < h):
            errors.append(f"station_pool[{i}] origin ({x}, {z}) out of bounds ({w}x{h})")
        if not (0 <= x + dx <= w and 0 <= z + dz <= h):
            errors.append(f"station_pool[{i}] extends beyond map bounds")

    return errors


def validate_map_file(path: Path) -> list[str]:
    errors = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return [f"Failed to parse: {e}"]

    required = [
        "schema_version", "map_id", "size", "custom_map",
        "agent_spawn_pool", "npc_spawn_pool", "station_pool",
        "max_battery", "max_steps", "hero_id", "local_view_size",
    ]
    for key in required:
        if key not in data:
            errors.append(f"Missing required field: {key}")

    if data.get("schema_version") != 1:
        errors.append(f"Unexpected schema_version: {data.get('schema_version')}")

    if errors:
        return errors

    errors += _check_dimensions(data)
    errors += _check_pools(data)

    return errors


def main():
    json_files = sorted(MAPS_DIR.glob("map_*.json"))

    if not json_files:
        print(f"No map_*.json files found in {MAPS_DIR}")
        sys.exit(1)

    print(f"Schema: {SCHEMA_FILE}")
    print(f"Validating {len(json_files)} map file(s)...\n")

    all_ok = True
    for path in json_files:
        errors = validate_map_file(path)
        if errors:
            all_ok = False
            print(f"  [FAIL] {path.name}")
            for e in errors:
                print(f"         - {e}")
        else:
            print(f"  [OK]   {path.name}")

    print()
    if all_ok:
        print("All maps passed validation.")
    else:
        print("Some maps have errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
