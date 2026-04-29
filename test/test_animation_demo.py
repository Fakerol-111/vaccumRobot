from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.simple_map_config import SIMPLE_MAP_CONFIG
from env import GridWorldEnv, TrajectoryRecorder


def test_trajectory_recorder_captures_charging_stations() -> None:
    recorder = TrajectoryRecorder()
    env = GridWorldEnv(**SIMPLE_MAP_CONFIG, enable_recording=True, trajectory_recorder=recorder, agent_spawn_mode=0)

    env.reset(seed=42)

    assert recorder.frames
    first_frame = recorder.frames[0]
    assert (first_frame.rendered_map == env.CHARGING_STATION).any()


def test_trajectory_animation_export() -> None:
    recorder = TrajectoryRecorder()
    env = GridWorldEnv(**SIMPLE_MAP_CONFIG, enable_recording=True, trajectory_recorder=recorder, agent_spawn_mode=0)

    payload = env.reset(seed=42)
    assert payload["frame_no"] == 0

    actions = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,3, 7, 1, 5, 2, 0, 4, 6, 1, 3, 5, 2, 7, 0, 4, 6, 2, 1, 5, 3, 7, 4, 0, 6, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0]
    for action in actions:
        payload = env.step(action)
        if payload["terminated"] or payload["truncated"]:
            break

    output_dir = PROJECT_ROOT / "artifacts" / "simple"
    output_dir.mkdir(parents=True, exist_ok=True)

    final_score = recorder.frames[-1].score
    total_steps = len(recorder.frames)
    base_name = f"score_{final_score}_step_{total_steps}"

    gif_path = recorder.export_gif(output_dir / f"{base_name}.gif", fps=10)
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0

    log_path = recorder.export_log(output_dir / f"{base_name}.log")
    assert log_path.exists()
    assert log_path.stat().st_size > 0

if __name__ == "__main__":
    test_trajectory_animation_export()