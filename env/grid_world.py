from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4
from typing import Any, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .trajectory_recorder import TrajectoryRecorder


@dataclass(frozen=True)
class NPC:
    """NPC configuration.

    Coordinates use (x, z), where x is the column index and z is the row index.
    """

    spawn: tuple[int, int]


@dataclass(frozen=True)
class ChargingStation:
    """Charging station rectangle configuration.

    Coordinates use (x, z).
    Width maps to dz and height maps to dx according to the project
    convention requested by the user.

    The station occupies [x, x + dx) in the horizontal direction and
    [z, z + dz) in the vertical direction on the grid.
    """

    x: int
    z: int
    dx: int
    dz: int


class GridWorldEnv(gym.Env):
    """A customizable grid-world environment based on Gymnasium.

    Base map convention:
    - 0: blocked / non-traversable
    - 1: traversable and clean
    - 2: traversable and dirty

    Overlay convention in rendered or observed map:
    - 3: NPC
    - 4: charging station
    - 5: agent

    The current implementation focuses on world construction, custom maps,
    NPC random movement inside a 21x21 area centered on its spawn point, and
    rectangular charging stations.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}

    BLOCKED = 0
    CLEAN = 1
    DIRTY = 2
    NPC_CELL = 3
    CHARGING_STATION = 4
    AGENT_CELL = 5

    ACTION_RIGHT = 0
    ACTION_RIGHT_UP = 1
    ACTION_UP = 2
    ACTION_LEFT_UP = 3
    ACTION_LEFT = 4
    ACTION_LEFT_DOWN = 5
    ACTION_DOWN = 6
    ACTION_RIGHT_DOWN = 7

    _ACTION_TO_DELTA = {
        ACTION_RIGHT: (1, 0),
        ACTION_RIGHT_UP: (1, -1),
        ACTION_UP: (0, -1),
        ACTION_LEFT_UP: (-1, -1),
        ACTION_LEFT: (-1, 0),
        ACTION_LEFT_DOWN: (-1, 1),
        ACTION_DOWN: (0, 1),
        ACTION_RIGHT_DOWN: (1, 1),
    }

    def __init__(
        self,
        size: tuple[int, int] = (128, 128),
        custom_map: np.ndarray | Iterable[Iterable[int]] | None = None,
        npcs: Iterable[NPC | dict[str, Any] | tuple[int, int]] | None = None,
        charging_stations: Iterable[ChargingStation | dict[str, int] | tuple[int, int, int, int]] | None = None,
        agent_spawn: tuple[int, int] | None = None,
        agent_position: tuple[int, int] | None = None,
        npc_walk_radius: int = 10,
        max_battery: int = 100,
        max_steps: int = 1000,
        hero_id: int = 37,
        npc_ids: Iterable[int] | None = None,
        map_id: int = 0,
        local_view_size: int = 21,
        enable_recording: bool = False,
        trajectory_recorder: TrajectoryRecorder | None = None,
        render_mode: str | None = None,
        agent_spawn_pool: Iterable[tuple[int, int]] | None = None,
        agent_spawn_mode: int = -1,
        npc_spawn_pool: Iterable[tuple[int, int]] | None = None,
        npc_spawn_modes: Iterable[int] | None = None,
        npc_count: int = 1,
        station_pool: Iterable[dict[str, int]] | None = None,
        station_count: int = 4,
        station_mode: int | Iterable[int] = -1,
    ) -> None:
        super().__init__()

        self.width, self.height = self._validate_size(size)
        self.npc_walk_radius = self._validate_npc_walk_radius(npc_walk_radius)
        self.max_battery = self._validate_positive_int(max_battery, "max_battery")
        self.max_steps = self._validate_positive_int(max_steps, "max_steps")
        self.hero_id = int(hero_id)
        self.map_id = int(map_id)
        self.local_view_size = self._validate_positive_odd_int(local_view_size, "local_view_size")
        self.enable_recording = bool(enable_recording)
        self.trajectory_recorder = trajectory_recorder if trajectory_recorder is not None else TrajectoryRecorder()
        self.render_mode = render_mode

        self._agent_spawn_pool = list(agent_spawn_pool) if agent_spawn_pool is not None else None
        self._agent_spawn_mode = int(agent_spawn_mode)
        self._npc_spawn_pool = list(npc_spawn_pool) if npc_spawn_pool is not None else None
        self._npc_spawn_modes = list(npc_spawn_modes) if npc_spawn_modes is not None else None
        self._npc_count = int(npc_count)
        self._station_pool = list(station_pool) if station_pool is not None else None
        self._station_count = int(station_count)
        self._station_mode = list(station_mode) if isinstance(station_mode, (list, tuple)) else int(station_mode)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}, got {render_mode!r}.")

        self.base_map = self._build_base_map(custom_map)
        self.initial_base_map = self.base_map.copy()
        self.initial_dirty_count = int(np.count_nonzero(self.base_map == self.DIRTY))

        if self._npc_spawn_pool is not None:
            self._resolve_npc_spawns()
        else:
            self.npcs = self._normalize_npcs(npcs)
        self.npc_ids = self._normalize_npc_ids(npc_ids)

        if self._station_pool is not None:
            self._resolve_station_spawns()
        else:
            self.charging_stations = self._normalize_charging_stations(charging_stations)

        if self._agent_spawn_pool is not None:
            self._resolve_agent_spawn()
        else:
            self.agent_spawn = self._normalize_agent_position(agent_spawn=agent_spawn, agent_position=agent_position)

        self._validate_agent_spawn()
        self._validate_npc_spawns()
        self._validate_charging_stations()

        self.agent_position = self.agent_spawn
        self.agent_battery = self.max_battery
        self.steps_taken = 0
        self.score = 0
        self.dirt_cleaned = 0
        self.charge_count = 0
        self.last_cleaned_cells: list[tuple[int, int]] = []
        self.env_id = self._generate_env_id()
        self.npc_positions: list[tuple[int, int]] = []
        self.current_map = self.base_map.copy()
        self._mode = "train"

        self.action_space = spaces.Discrete(len(self._ACTION_TO_DELTA))
        self.observation_space = spaces.Dict(
            {
                "map": spaces.Box(
                    low=0,
                    high=max(self.DIRTY, self.NPC_CELL, self.CHARGING_STATION, self.AGENT_CELL),
                    shape=(self.height, self.width),
                    dtype=np.int8,
                ),
                "agent_position": spaces.Box(
                    low=0,
                    high=max(self.width - 1, self.height - 1),
                    shape=(2,),
                    dtype=np.int32,
                ),
                "agent_battery": spaces.Box(
                    low=0,
                    high=self.max_battery,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "npc_positions": spaces.Box(
                    low=0,
                    high=max(self.width - 1, self.height - 1),
                    shape=(len(self.npcs), 2),
                    dtype=np.int32,
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        super().reset(seed=seed)
        if options and "mode" in options:
            mode = options["mode"]
            if mode not in ("train", "eval"):
                raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}.")
            self._mode = mode

        if self._agent_spawn_pool is not None:
            self._resolve_agent_spawn()
        if self._npc_spawn_pool is not None:
            self._resolve_npc_spawns()
        if self._station_pool is not None:
            self._resolve_station_spawns()

        self.env_id = self._generate_env_id()
        self.agent_position = self.agent_spawn
        self.agent_battery = self.max_battery
        self.steps_taken = 0
        self.score = 0
        self.dirt_cleaned = 0
        self.charge_count = 0
        self.last_cleaned_cells = []
        self.npc_positions = [npc.spawn for npc in self.npcs]
        self.base_map = self.initial_base_map.copy()
        self.initial_dirty_count = int(np.count_nonzero(self.base_map == self.DIRTY))
        self._clean_current_positions()
        self._recharge_if_on_station(increment_counter=False)
        self._refresh_current_map()
        near_npc = self._is_agent_near_any_npc()
        battery_depleted = self.agent_battery == 0
        terminated = near_npc or battery_depleted
        truncated = self.steps_taken >= self.max_steps
        self._record_current_frame(terminated=terminated, truncated=truncated, reset=True)
        return self._build_return_payload(
            terminated=terminated,
            truncated=truncated,
            near_npc=near_npc,
            battery_depleted=battery_depleted,
        )

    def step(self, action: int) -> dict[str, Any]:
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer in [0, {self.action_space.n - 1}], got {action!r}.")

        self.last_cleaned_cells = []
        self.steps_taken += 1
        self.agent_position = self._compute_candidate_position(self.agent_position, self._ACTION_TO_DELTA[action])
        self.agent_battery = max(0, self.agent_battery - 1)
        self._clean_agent_position(self.agent_position)
        self._recharge_if_on_station()
        self._move_npcs()
        self._clean_positions(self.npc_positions)
        self._refresh_current_map()

        near_npc = self._is_agent_near_any_npc()
        battery_depleted = self.agent_battery == 0
        terminated = near_npc or battery_depleted
        truncated = self.steps_taken >= self.max_steps
        self._record_current_frame(terminated=terminated, truncated=truncated, reset=False)
        return self._build_return_payload(
            terminated=terminated,
            truncated=truncated,
            near_npc=near_npc,
            battery_depleted=battery_depleted,
        )

    def render(self) -> str | np.ndarray | None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def close(self) -> None:
        return None

    def _build_base_map(self, custom_map: np.ndarray | Iterable[Iterable[int]] | None) -> np.ndarray:
        if custom_map is None:
            return np.ones((self.height, self.width), dtype=np.int8)

        grid = np.asarray(custom_map, dtype=np.int8)
        expected_shape = (self.height, self.width)
        if grid.shape != expected_shape:
            raise ValueError(f"custom_map shape must be {expected_shape}, got {grid.shape}.")
        if np.any((grid < self.BLOCKED) | (grid > self.DIRTY)):
            raise ValueError("custom_map values must follow the convention: 0=blocked, 1=clean, 2=dirty.")
        return grid.copy()

    def _normalize_npcs(self, npcs: Iterable[NPC | dict[str, Any] | tuple[int, int]] | None) -> list[NPC]:
        if npcs is None:
            return []

        normalized: list[NPC] = []
        for npc in npcs:
            if isinstance(npc, NPC):
                normalized.append(npc)
            elif isinstance(npc, dict):
                spawn = npc.get("spawn")
                if spawn is None:
                    raise ValueError("NPC dict must contain a 'spawn' field.")
                normalized.append(NPC(spawn=self._to_position_tuple(spawn, "NPC spawn")))
            else:
                normalized.append(NPC(spawn=self._to_position_tuple(npc, "NPC spawn")))
        return normalized

    def _normalize_charging_stations(
        self,
        charging_stations: Iterable[ChargingStation | dict[str, int] | tuple[int, int, int, int]] | None,
    ) -> list[ChargingStation]:
        if charging_stations is None:
            return []

        normalized: list[ChargingStation] = []
        for station in charging_stations:
            if isinstance(station, ChargingStation):
                normalized.append(station)
            elif isinstance(station, dict):
                x = int(station["x"])
                z = int(station["z"])
                dx = int(station.get("dx", station.get("height")))
                dz = int(station.get("dz", station.get("width")))
                normalized.append(ChargingStation(x=x, z=z, dx=dx, dz=dz))
            else:
                if len(station) != 4:
                    raise ValueError("Charging station tuple must be (x, z, dx, dz).")
                x, z, dx, dz = station
                normalized.append(ChargingStation(x=int(x), z=int(z), dx=int(dx), dz=int(dz)))
        return normalized

    def _normalize_agent_position(
        self,
        agent_spawn: tuple[int, int] | None,
        agent_position: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if agent_spawn is not None and agent_position is not None:
            raise ValueError("Use either agent_spawn or agent_position, not both.")
        if agent_position is not None:
            return self._to_position_tuple(agent_position, "agent position")
        if agent_spawn is not None:
            return self._to_position_tuple(agent_spawn, "agent spawn")
        return 0, 0

    def _resolve_agent_spawn(self) -> None:
        pool = self._agent_spawn_pool
        mode = self._agent_spawn_mode
        if mode == -1:
            idx = int(self.np_random.integers(len(pool)))
        else:
            idx = int(mode)
        self.agent_spawn = tuple(pool[idx])

    def _resolve_npc_spawns(self) -> None:
        pool = self._npc_spawn_pool
        modes = self._npc_spawn_modes or [-1] * self._npc_count
        used: dict[int, int] = {}
        new_npcs: list[NPC] = []
        for i, mode in enumerate(modes):
            if mode == -1:
                available = [j for j in range(len(pool)) if j not in used]
                if not available:
                    available = list(range(len(pool)))
                    used.clear()
                idx = int(self.np_random.integers(len(available)))
                idx = available[idx]
            else:
                idx = int(mode)
            used[idx] = i
            new_npcs.append(NPC(spawn=tuple(pool[idx])))
        self.npcs = new_npcs

    def _resolve_station_spawns(self) -> None:
        pool = self._station_pool
        mode = self._station_mode
        count = self._station_count
        if isinstance(mode, int) and mode == -1:
            indices = self.np_random.choice(len(pool), size=count, replace=False).tolist()
            indices = [int(i) for i in indices]
        elif isinstance(mode, (list, tuple)):
            indices = [int(m) for m in mode]
        else:
            indices = [int(mode)]
        self.charging_stations = [
            self._single_station(pool[i]) for i in indices
        ]

    @staticmethod
    def _single_station(cfg: dict[str, int]) -> ChargingStation:
        return ChargingStation(
            x=int(cfg["x"]),
            z=int(cfg["z"]),
            dx=int(cfg.get("dx", cfg.get("height", 3))),
            dz=int(cfg.get("dz", cfg.get("width", 3))),
        )

    def _validate_agent_spawn(self) -> None:
        x, z = self.agent_spawn
        if not self._is_in_bounds(x, z):
            raise ValueError(f"agent spawn {self.agent_spawn} is outside the grid.")
        if self.base_map[z, x] == self.BLOCKED:
            raise ValueError(f"agent spawn {self.agent_spawn} must be on a traversable cell.")

    def _validate_npc_spawns(self) -> None:
        for npc in self.npcs:
            x, z = npc.spawn
            if not self._is_in_bounds(x, z):
                raise ValueError(f"NPC spawn {npc.spawn} is outside the grid.")
            if self.base_map[z, x] == self.BLOCKED:
                raise ValueError(f"NPC spawn {npc.spawn} must be on a traversable cell.")

    def _validate_charging_stations(self) -> None:
        for station in self.charging_stations:
            if station.dx <= 0 or station.dz <= 0:
                raise ValueError("Charging station dx and dz must be positive.")
            if not self._is_in_bounds(station.x, station.z):
                raise ValueError(f"Charging station origin {(station.x, station.z)} is outside the grid.")
            if station.x + station.dx > self.width or station.z + station.dz > self.height:
                raise ValueError(f"Charging station {station} exceeds grid bounds.")

    def _move_npcs(self) -> None:
        if not self.npc_positions:
            return

        occupied = set(self.npc_positions)
        next_positions: list[tuple[int, int]] = []

        for npc, current_position in zip(self.npcs, self.npc_positions, strict=True):
            occupied.discard(current_position)
            candidates = self._valid_npc_moves(npc.spawn, current_position, occupied)
            selected_index = int(self.np_random.integers(len(candidates)))
            next_position = candidates[selected_index]
            next_positions.append(next_position)
            occupied.add(next_position)

        self.npc_positions = next_positions

    def _clean_current_positions(self) -> None:
        self._clean_agent_position(self.agent_position)
        self._clean_positions(self.npc_positions)

    def _record_current_frame(self, *, terminated: bool, truncated: bool, reset: bool) -> None:
        if not self.enable_recording:
            return
        if reset:
            self.trajectory_recorder.clear()
        self.trajectory_recorder.record(
            step_no=self.steps_taken,
            rendered_map=self.current_map,
            agent_position=self.agent_position,
            npc_positions=list(self.npc_positions),
            battery=self.agent_battery,
            score=self.score,
            terminated=terminated,
            truncated=truncated,
        )

    def _build_return_payload(
        self,
        terminated: bool,
        truncated: bool,
        near_npc: bool = False,
        battery_depleted: bool = False,
    ) -> dict[str, Any]:
        frame_no = self.steps_taken
        frame_state = self._build_frame_state(include_hidden_npcs=False)
        observation = {
            "step_no": self.steps_taken,
            "frame_state": frame_state,
            "env_info": self._build_env_info(),
            "map_info": self._get_local_map_info(),
            "legal_action": self._get_legal_action_mask(),
        }
        result_code = 0
        result_message = ""
        if terminated:
            if battery_depleted:
                result_code = 1
                result_message = "battery_depleted"
            elif near_npc:
                result_code = 2
                result_message = "caught_by_npc"
        elif truncated:
            result_code = 3
            result_message = "max_steps_reached"
        extra_info: dict[str, Any] = {}
        if self._mode == "train":
            extra_info = {
                "frame_state": self._build_frame_state(include_hidden_npcs=True, include_frame_no=True),
                "map_id": self.map_id,
                "result_code": result_code,
                "result_message": result_message,
            }
        return {
            "env_id": self.env_id,
            "frame_no": frame_no,
            "observation": observation,
            "extra_info": extra_info,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _build_frame_state(self, include_hidden_npcs: bool, include_frame_no: bool = False) -> dict[str, Any]:
        frame_state: dict[str, Any] = {
            "heroes": {
                "battery": self.agent_battery,
                "battery_max": self.max_battery,
                "dirt_cleaned": self.dirt_cleaned,
                "hero_id": self.hero_id,
                "pos": self._build_pos_dict(self.agent_position),
                "score": self.score,
            },
            "organs": self._build_organs_info(),
            "npcs": self._build_npcs_info(include_hidden_npcs=include_hidden_npcs),
        }
        if include_frame_no:
            frame_state["frame_no"] = self.steps_taken
        return frame_state

    def _build_env_info(self) -> dict[str, Any]:
        return {
            "battery_max": self.max_battery,
            "charge_count": self.charge_count,
            "clean_score": self.score,
            "max_step": self.max_steps,
            "npc_count": len(self.npcs),
            "pos": self._build_pos_dict(self.agent_position),
            "remaining_charge": self.agent_battery,
            "step_cleaned_cells": [self._build_pos_dict(position) for position in self.last_cleaned_cells],
            "step_no": self.steps_taken,
            "total_charger": len(self.charging_stations),
            "total_dirt": self.initial_dirty_count,
            "total_score": self.score,
        }

    def _build_organs_info(self) -> list[dict[str, Any]]:
        organs: list[dict[str, Any]] = []
        for idx, station in enumerate(self.charging_stations, start=1):
            organs.append(
                {
                    "config_id": idx,
                    "h": station.dx,
                    "pos": {"x": station.x, "z": station.z},
                    "sub_type": 1,
                    "w": station.dz,
                }
            )
        return organs

    def _build_npcs_info(self, include_hidden_npcs: bool) -> list[dict[str, Any]]:
        npcs_info: list[dict[str, Any]] = []
        for idx, (npc_id, position) in enumerate(zip(self.npc_ids, self.npc_positions, strict=True), start=1):
            visible = self._is_position_in_local_view(position)
            if include_hidden_npcs:
                npcs_info.append({"idx": idx, "npc_id": npc_id, "pos": self._build_pos_dict(position)})
            else:
                npcs_info.append(
                    {
                        "npc_id": npc_id,
                        "is_in_view": int(visible),
                        "pos": self._build_pos_dict(position) if visible else {"x": -1, "z": -1},
                    }
                )
        return npcs_info

    def _get_local_map_info(self) -> list[list[int]]:
        radius = self.local_view_size // 2
        center_x, center_z = self.agent_position
        rows: list[list[int]] = []
        for z in range(center_z - radius, center_z + radius + 1):
            row: list[int] = []
            for x in range(center_x - radius, center_x + radius + 1):
                if self._is_in_bounds(x, z):
                    row.append(int(self.base_map[z, x]))
                else:
                    row.append(self.BLOCKED)
            rows.append(row)
        return rows

    def _get_legal_action_mask(self) -> list[int]:
        mask: list[int] = []
        for action in range(self.action_space.n):
            next_position = self._compute_candidate_position(self.agent_position, self._ACTION_TO_DELTA[action])
            mask.append(int(next_position != self.agent_position))
        return mask

    def _is_position_in_local_view(self, position: tuple[int, int]) -> bool:
        radius = self.local_view_size // 2
        return (
            abs(position[0] - self.agent_position[0]) <= radius
            and abs(position[1] - self.agent_position[1]) <= radius
        )

    def _build_pos_dict(self, position: tuple[int, int]) -> dict[str, int]:
        return {"x": int(position[0]), "z": int(position[1])}

    def _generate_env_id(self) -> str:
        return str(uuid4().int)[:8]

    def _normalize_npc_ids(self, npc_ids: Iterable[int] | None) -> list[int]:
        if npc_ids is None:
            return [38 + idx for idx in range(len(self.npcs))]

        normalized = [int(npc_id) for npc_id in npc_ids]
        if len(normalized) != len(self.npcs):
            raise ValueError("npc_ids length must match npc count.")
        return normalized

    def _recharge_if_on_station(self, increment_counter: bool = True) -> None:
        if self._is_in_charging_station(self.agent_position):
            if increment_counter and self.agent_battery < self.max_battery:
                self.charge_count += 1
            self.agent_battery = self.max_battery

    def _is_in_charging_station(self, position: tuple[int, int]) -> bool:
        x, z = position
        for station in self.charging_stations:
            if station.x <= x < station.x + station.dx and station.z <= z < station.z + station.dz:
                return True
        return False

    def _is_agent_near_any_npc(self) -> bool:
        for npc_x, npc_z in self.npc_positions:
            if abs(self.agent_position[0] - npc_x) <= 1 and abs(self.agent_position[1] - npc_z) <= 1:
                return True
        return False

    def _clean_positions(self, positions: Iterable[tuple[int, int]]) -> None:
        for position in positions:
            self._clean_position(position, award_score=False)

    def _clean_agent_position(self, position: tuple[int, int]) -> None:
        self._clean_position(position, award_score=True)

    def _clean_position(self, position: tuple[int, int], award_score: bool) -> None:
        x, z = position
        if self.base_map[z, x] == self.DIRTY:
            self.base_map[z, x] = self.CLEAN
            cell = (x, z)
            if cell not in self.last_cleaned_cells:
                self.last_cleaned_cells.append(cell)
            self.dirt_cleaned += 1
            if award_score:
                self.score += 1

    def _valid_npc_moves(
        self,
        spawn: tuple[int, int],
        current_position: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        for dx, dz in self._ACTION_TO_DELTA.values():
            next_position = self._compute_candidate_position(current_position, (dx, dz))
            if next_position == current_position:
                continue
            x, z = next_position
            if not self._is_inside_npc_walk_area(spawn, next_position):
                continue
            if (x, z) in occupied:
                continue
            candidates.append(next_position)
        return candidates or [current_position]

    def _compute_candidate_position(
        self,
        current_position: tuple[int, int],
        delta: tuple[int, int],
    ) -> tuple[int, int]:
        dx, dz = delta
        x = current_position[0] + dx
        z = current_position[1] + dz

        if not self._is_in_bounds(x, z):
            return current_position
        if self.base_map[z, x] == self.BLOCKED:
            return current_position
        if dx != 0 and dz != 0 and not self._can_move_diagonally(current_position, dx, dz):
            return current_position
        return x, z

    def _can_move_diagonally(self, current_position: tuple[int, int], dx: int, dz: int) -> bool:
        side_positions = [
            (current_position[0] + dx, current_position[1]),
            (current_position[0], current_position[1] + dz),
        ]

        for x, z in side_positions:
            if self._is_in_bounds(x, z) and self.base_map[z, x] != self.BLOCKED:
                return True
        return False

    def _is_inside_npc_walk_area(self, spawn: tuple[int, int], position: tuple[int, int]) -> bool:
        return (
            abs(position[0] - spawn[0]) <= self.npc_walk_radius
            and abs(position[1] - spawn[1]) <= self.npc_walk_radius
        )

    def _refresh_current_map(self) -> None:
        self.current_map = self.base_map.copy()

        for station in self.charging_stations:
            self.current_map[station.z : station.z + station.dz, station.x : station.x + station.dx] = self.CHARGING_STATION

        for x, z in self.npc_positions:
            self.current_map[z, x] = self.NPC_CELL

        agent_x, agent_z = self.agent_position
        self.current_map[agent_z, agent_x] = self.AGENT_CELL

    def _get_observation(self) -> dict[str, np.ndarray]:
        if self.npc_positions:
            npc_positions = np.asarray(self.npc_positions, dtype=np.int32)
        else:
            npc_positions = np.empty((0, 2), dtype=np.int32)
        return {
            "map": self.current_map.copy(),
            "agent_position": np.asarray(self.agent_position, dtype=np.int32),
            "agent_battery": np.asarray([self.agent_battery], dtype=np.int32),
            "npc_positions": npc_positions,
        }

    def _get_info(self) -> dict[str, Any]:
        near_npc = self._is_agent_near_any_npc()
        battery_depleted = self.agent_battery == 0
        terminated = near_npc or battery_depleted
        truncated = self.steps_taken >= self.max_steps
        return self._build_return_payload(
            terminated=terminated,
            truncated=truncated,
            near_npc=near_npc,
            battery_depleted=battery_depleted,
        )

    def _render_ansi(self) -> str:
        symbols = {
            self.BLOCKED: "#",
            self.CLEAN: ".",
            self.DIRTY: "*",
            self.NPC_CELL: "N",
            self.CHARGING_STATION: "C",
            self.AGENT_CELL: "A",
        }
        rows = []
        for row in self.current_map:
            rows.append("".join(symbols.get(int(cell), "?") for cell in row))
        return "\n".join(rows)

    def _render_rgb_array(self) -> np.ndarray:
        colors = np.array(
            [
                [30, 30, 30],
                [255, 255, 255],
                [205, 170, 60],
                [220, 60, 60],
                [60, 140, 240],
                [80, 220, 120],
            ],
            dtype=np.uint8,
        )
        clipped = np.clip(self.current_map, 0, len(colors) - 1)
        return colors[clipped]

    def _is_in_bounds(self, x: int, z: int) -> bool:
        return 0 <= x < self.width and 0 <= z < self.height

    @staticmethod
    def _validate_size(size: tuple[int, int]) -> tuple[int, int]:
        if len(size) != 2:
            raise ValueError("size must be a tuple of (width, height).")
        width, height = int(size[0]), int(size[1])
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive.")
        return width, height

    @staticmethod
    def _validate_npc_walk_radius(npc_walk_radius: int) -> int:
        radius = int(npc_walk_radius)
        if radius < 0:
            raise ValueError("npc_walk_radius cannot be negative.")
        return radius

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        normalized = int(value)
        if normalized <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        return normalized

    @staticmethod
    def _validate_positive_odd_int(value: int, name: str) -> int:
        normalized = int(value)
        if normalized <= 0 or normalized % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer.")
        return normalized

    @staticmethod
    def _to_position_tuple(value: Any, label: str) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError(f"{label} must contain exactly two coordinates: (x, z).")
        return int(value[0]), int(value[1])


