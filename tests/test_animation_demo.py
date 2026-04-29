from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.maps.src.map_1 import MAP_CONFIG, build_map
from env import GridWorldEnv, TrajectoryRecorder


class TestAnimationDemo(unittest.TestCase):
    def test_trajectory_recorder_captures_charging_stations(self):
        recorder = TrajectoryRecorder()
        cfg = dict(MAP_CONFIG)
        cfg["custom_map"] = build_map(128)
        cfg["npc_count"] = 1
        cfg["station_count"] = 4
        env = GridWorldEnv(**cfg, enable_recording=True, trajectory_recorder=recorder, agent_spawn_mode=0)
        env.reset(seed=42)
        self.assertTrue(recorder.frames)
        first_frame = recorder.frames[0]
        self.assertTrue((first_frame.rendered_map == env.CHARGING_STATION).any())

    def test_trajectory_animation_export(self):
        recorder = TrajectoryRecorder()
        cfg = dict(MAP_CONFIG)
        cfg["custom_map"] = build_map(128)
        cfg["npc_count"] = 1
        cfg["station_count"] = 4
        env = GridWorldEnv(**cfg, enable_recording=True, trajectory_recorder=recorder, agent_spawn_mode=0)
        payload = env.reset(seed=42)
        self.assertEqual(payload["frame_no"], 0)

        actions = [6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,3, 7, 1, 5, 2, 0, 4, 6, 1, 3, 5, 2, 7, 0, 4, 6, 2, 1, 5, 3, 7, 4, 0, 6, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0, 2, 5, 1, 3, 7, 4, 6, 0]
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
        self.assertTrue(gif_path.exists())
        self.assertGreater(gif_path.stat().st_size, 0)

        log_path = recorder.export_log(output_dir / f"{base_name}.log")
        self.assertTrue(log_path.exists())
        self.assertGreater(log_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
