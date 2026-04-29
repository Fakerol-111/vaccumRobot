from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.maps.src.map_1 import MAP_CONFIG, build_map
from env import GridWorldEnv


class TestSimpleMap(unittest.TestCase):
    def test_layout(self):
        grid = build_map(128)
        self.assertEqual(grid.shape, (128, 128))
        self.assertTrue((grid[0, :] == 0).all())
        self.assertTrue((grid[-1, :] == 0).all())
        self.assertTrue((grid[:, 0] == 0).all())
        self.assertTrue((grid[:, -1] == 0).all())
        self.assertTrue((grid[1:-1, 1:-1] == 2).all())

    def test_config_contents(self):
        self.assertEqual(MAP_CONFIG["size"], (128, 128))
        self.assertEqual(len(MAP_CONFIG["agent_spawn_pool"]), 4)
        self.assertEqual(len(MAP_CONFIG["npc_spawn_pool"]), 5)
        self.assertEqual(len(MAP_CONFIG["station_pool"]), 4)

    def test_environment_can_be_created(self):
        cfg = dict(MAP_CONFIG)
        cfg["custom_map"] = build_map(128)
        cfg["npc_count"] = 1
        cfg["station_count"] = 4
        env = GridWorldEnv(**cfg)
        payload = env.reset(seed=42)
        self.assertTrue(payload["env_id"])
        self.assertEqual(payload["frame_no"], 0)
        self.assertEqual(payload["observation"]["frame_state"]["heroes"]["battery_max"], 200)
        self.assertEqual(len(payload["observation"]["frame_state"]["organs"]), 4)
        self.assertEqual(len(payload["extra_info"]["frame_state"]["npcs"]), 1)

    def test_environment_with_fixed_spawns(self):
        cfg = dict(MAP_CONFIG)
        cfg["custom_map"] = build_map(128)
        cfg["npc_count"] = 1
        cfg["station_count"] = 4
        env = GridWorldEnv(**cfg, agent_spawn_mode=0, npc_spawn_modes=[4])
        payload = env.reset(seed=42)
        self.assertEqual(payload["observation"]["frame_state"]["heroes"]["pos"], {"x": 64, "z": 1})
        npc_pos = payload["extra_info"]["frame_state"]["npcs"][0]["pos"]
        self.assertEqual(npc_pos, {"x": 96, "z": 32})

    def test_random_spawns_differ_per_reset(self):
        cfg = dict(MAP_CONFIG)
        cfg["custom_map"] = build_map(128)
        cfg["npc_count"] = 1
        cfg["station_count"] = 4
        env = GridWorldEnv(**cfg)
        positions = set()
        for seed_val in range(20):
            payload = env.reset(seed=seed_val)
            pos = (
                payload["observation"]["frame_state"]["heroes"]["pos"]["x"],
                payload["observation"]["frame_state"]["heroes"]["pos"]["z"],
            )
            positions.add(pos)
        self.assertGreater(len(positions), 1)

    def test_charging_stations_are_rendered_on_map(self):
        cfg = dict(MAP_CONFIG)
        cfg["custom_map"] = build_map(128)
        cfg["npc_count"] = 1
        cfg["station_count"] = 4
        env = GridWorldEnv(**cfg, agent_spawn_mode=0)
        env.reset(seed=42)
        for station in env.charging_stations:
            station_area = env.current_map[station.z : station.z + station.dz, station.x : station.x + station.dx]
            self.assertEqual(station_area.shape, (station.dz, station.dx))
            self.assertTrue((station_area == env.CHARGING_STATION).all())


if __name__ == "__main__":
    unittest.main()
