from __future__ import annotations

from typing import Any, TYPE_CHECKING

from env.grid_world import GridWorldEnv

if TYPE_CHECKING:
    from env.trajectory_recorder import TrajectoryRecorder

_KNOWN_ENV_KEYS = {
    "size", "custom_map", "npcs", "charging_stations",
    "agent_spawn", "agent_position", "npc_walk_radius",
    "max_battery", "max_steps", "hero_id", "npc_ids", "map_id",
    "local_view_size", "agent_spawn_pool", "agent_spawn_mode",
    "npc_spawn_pool", "npc_spawn_modes", "npc_count",
    "station_pool", "station_count", "station_mode",
}


def create_env(
    map_config: dict[str, Any],
    *,
    enable_recording: bool = False,
    trajectory_recorder: TrajectoryRecorder | None = None,
    render_mode: str | None = None,
) -> GridWorldEnv:
    kwargs = {k: v for k, v in map_config.items() if k in _KNOWN_ENV_KEYS}
    return GridWorldEnv(
        **kwargs,
        enable_recording=enable_recording,
        trajectory_recorder=trajectory_recorder,
        render_mode=render_mode,
    )
