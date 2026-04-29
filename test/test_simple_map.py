from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.simple_map_config import SIMPLE_MAP_CONFIG, build_simple_map
from env import GridWorldEnv


def test_simple_map_layout() -> None:
    grid = build_simple_map(128)

    assert grid.shape == (128, 128)
    assert (grid[0, :] == 0).all()
    assert (grid[-1, :] == 0).all()
    assert (grid[:, 0] == 0).all()
    assert (grid[:, -1] == 0).all()
    assert (grid[1:-1, 1:-1] == 2).all()


def test_simple_map_config_contents() -> None:
    config = SIMPLE_MAP_CONFIG

    assert config["size"] == (128, 128)
    assert len(config["agent_spawn_pool"]) == 4
    assert len(config["npc_spawn_pool"]) == 5
    assert len(config["station_pool"]) == 4


def test_environment_can_be_created_from_config() -> None:
    env = GridWorldEnv(**SIMPLE_MAP_CONFIG)
    payload = env.reset(seed=42)

    assert payload["env_id"]
    assert payload["frame_no"] == 0
    assert payload["observation"]["frame_state"]["heroes"]["battery_max"] == 200
    assert len(payload["observation"]["frame_state"]["organs"]) == 4
    assert len(payload["extra_info"]["frame_state"]["npcs"]) == 1

    print(payload)


def test_environment_with_fixed_spawns() -> None:
    env = GridWorldEnv(
        **SIMPLE_MAP_CONFIG,
        agent_spawn_mode=0,
        npc_spawn_modes=[4],
    )
    payload = env.reset(seed=42)

    assert payload["observation"]["frame_state"]["heroes"]["pos"] == {"x": 64, "z": 1}
    npc_pos = payload["extra_info"]["frame_state"]["npcs"][0]["pos"]
    assert npc_pos == {"x": 96, "z": 32}


def test_environment_random_spawns_differ_per_reset() -> None:
    env = GridWorldEnv(**SIMPLE_MAP_CONFIG)

    positions = set()
    for seed_val in range(20):
        payload = env.reset(seed=seed_val)
        pos = (
            payload["observation"]["frame_state"]["heroes"]["pos"]["x"],
            payload["observation"]["frame_state"]["heroes"]["pos"]["z"],
        )
        positions.add(pos)

    assert len(positions) > 1


def test_charging_stations_are_rendered_on_map() -> None:
    env = GridWorldEnv(**SIMPLE_MAP_CONFIG, agent_spawn_mode=0)
    env.reset(seed=42)

    for station in env.charging_stations:
        station_area = env.current_map[station.z : station.z + station.dz, station.x : station.x + station.dx]
        assert station_area.shape == (station.dz, station.dx)
        assert (station_area == env.CHARGING_STATION).all()



if __name__ == "__main__":
    test_simple_map_layout()
    test_simple_map_config_contents()
    test_environment_can_be_created_from_config()
    test_environment_with_fixed_spawns()
    test_environment_random_spawns_differ_per_reset()
    test_charging_stations_are_rendered_on_map()
    print("All tests passed!")
