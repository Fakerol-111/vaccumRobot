"""地图加载层测试：configs/map_loader.py"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from configs.map_loader import load_map_config, load_map_configs, _parse_custom_map


class TestParseCustomMap(unittest.TestCase):
    def test_parse_3x3_grid(self):
        rows = ["010", "111", "020"]
        grid = _parse_custom_map(rows, 3)
        self.assertIsInstance(grid, np.ndarray)
        self.assertEqual(grid.shape, (3, 3))
        self.assertEqual(grid.dtype, np.int8)
        expected = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 2, 0],
        ], dtype=np.int8)
        np.testing.assert_array_equal(grid, expected)

    def test_parse_single_row(self):
        rows = ["012"]
        grid = _parse_custom_map(rows, 3)
        self.assertEqual(grid.shape, (1, 3))

    def test_row_width_mismatch_raises(self):
        rows = ["010", "11"]
        with self.assertRaises(ValueError) as ctx:
            _parse_custom_map(rows, 3)
        self.assertIn("row 1 length 2", str(ctx.exception))

    def test_invalid_char_raises(self):
        rows = ["01x"]
        with self.assertRaises(ValueError) as ctx:
            _parse_custom_map(rows, 3)
        self.assertIn("invalid character", str(ctx.exception))

    def test_empty_rows_parses_correctly(self):
        rows = []
        grid = _parse_custom_map(rows, 5)
        self.assertEqual(grid.shape, (0, 5))


class TestLoadMapConfig(unittest.TestCase):
    def test_load_map_1(self):
        cfg = load_map_config(1)
        self.assertIsInstance(cfg, dict)
        self.assertEqual(cfg["map_id"], 1)
        self.assertEqual(cfg["schema_version"], 1)
        self.assertIsInstance(cfg["size"], tuple)
        self.assertEqual(len(cfg["size"]), 2)

    def test_custom_map_is_numpy_array(self):
        cfg = load_map_config(1)
        cm = cfg["custom_map"]
        self.assertIsInstance(cm, np.ndarray)
        self.assertEqual(cm.dtype, np.int8)

    def test_map_shape_matches_size(self):
        cfg = load_map_config(1)
        h, w = cfg["size"]
        cm = cfg["custom_map"]
        self.assertEqual(cm.shape, (h, w))

    def test_map_values_in_range(self):
        cfg = load_map_config(1)
        cm = cfg["custom_map"]
        self.assertTrue(np.all((cm >= 0) & (cm <= 2)),
                        "map grid values must be 0, 1, or 2")

    def test_has_required_fields(self):
        cfg = load_map_config(1)
        for key in ("agent_spawn_pool", "npc_spawn_pool", "station_pool",
                    "max_battery", "max_steps", "hero_id"):
            self.assertIn(key, cfg, f"missing field: {key}")

    def test_load_all_4_maps(self):
        for mid in [1, 2, 3, 4]:
            with self.subTest(map_id=mid):
                cfg = load_map_config(mid)
                self.assertEqual(cfg["map_id"], mid)
                self.assertIsInstance(cfg["custom_map"], np.ndarray)

    def test_nonexistent_map_raises(self):
        with self.assertRaises(ValueError) as ctx:
            load_map_config(999)
        self.assertIn("not found", str(ctx.exception).lower())


class TestLoadMapConfigs(unittest.TestCase):
    def test_load_multiple_maps(self):
        configs = load_map_configs([1, 2, 3])
        self.assertEqual(len(configs), 3)
        for cfg in configs:
            self.assertIsInstance(cfg["custom_map"], np.ndarray)

    def test_empty_list(self):
        configs = load_map_configs([])
        self.assertEqual(configs, [])

    def test_map_ids_matches_output(self):
        configs = load_map_configs([3, 1])
        self.assertEqual(configs[0]["map_id"], 3)
        self.assertEqual(configs[1]["map_id"], 1)


if __name__ == "__main__":
    unittest.main()
