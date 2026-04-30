"""Checkpoint 序列化测试：agent/common/checkpoint.py"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.common.checkpoint import (
    Checkpoint,
    build_config_snapshot,
    capture_rng_state,
    restore_rng_state,
)


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_to_dict_and_from_dict(self):
        ckpt = Checkpoint(
            model_state_dict={"layer1.weight": "mock"},
            optimizer_state_dict={"param_groups": []},
            global_step=5000,
            episode_counter=42,
            current_map_idx=1,
            current_map_id=3,
            current_stage_name="stage_2",
            config_snapshot={"learning_rate": 0.0003},
            rng_state={"python": "mock_state"},
        )
        data = ckpt.to_dict()
        restored = Checkpoint.from_dict(data)

        self.assertEqual(restored.model_state_dict, {"layer1.weight": "mock"})
        self.assertEqual(restored.optimizer_state_dict, {"param_groups": []})
        self.assertEqual(restored.global_step, 5000)
        self.assertEqual(restored.episode_counter, 42)
        self.assertEqual(restored.current_map_idx, 1)
        self.assertEqual(restored.current_map_id, 3)
        self.assertEqual(restored.current_stage_name, "stage_2")
        self.assertEqual(restored.config_snapshot, {"learning_rate": 0.0003})
        self.assertEqual(restored.rng_state, {"python": "mock_state"})
        self.assertEqual(restored.format_version, data["format_version"])

    def test_from_dict_minimal(self):
        data = {
            "model_state_dict": {"w": "v"},
            "optimizer_state_dict": {},
            "global_step": 100,
        }
        ckpt = Checkpoint.from_dict(data)
        self.assertEqual(ckpt.global_step, 100)
        self.assertEqual(ckpt.episode_counter, 0)
        self.assertEqual(ckpt.current_map_idx, 0)
        self.assertEqual(ckpt.current_map_id, 0)
        self.assertEqual(ckpt.current_stage_name, "")
        self.assertEqual(ckpt.config_snapshot, {})
        self.assertEqual(ckpt.rng_state, {})

    def test_from_dict_default_format_version(self):
        data = {
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "global_step": 0,
        }
        ckpt = Checkpoint.from_dict(data)
        self.assertEqual(ckpt.format_version, "1.0")

    def test_rng_state_capture_restore(self):
        state = capture_rng_state()
        self.assertIn("python", state)
        self.assertIn("numpy", state)
        self.assertIn("torch", state)
        # restore should not raise
        restore_rng_state(state)

    def test_rng_state_restore_empty(self):
        restore_rng_state({})

    def test_build_config_snapshot(self):
        config = SimpleNamespace(
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            ppo_epochs=4,
            batch_size=2048,
            mini_batch_size=256,
            total_timesteps=100000,
            save_interval=10000,
            log_interval=100,
            max_npcs=5,
            local_view_size=21,
            num_actions=6,
        )
        snapshot = build_config_snapshot(config)
        self.assertEqual(snapshot["learning_rate"], 0.0003)
        self.assertEqual(snapshot["batch_size"], 2048)
        self.assertEqual(snapshot["num_actions"], 6)

    def test_build_config_snapshot_with_extra(self):
        config = SimpleNamespace(
            learning_rate=0.0003, gamma=0.99, gae_lambda=0.95,
            clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
            max_grad_norm=0.5, ppo_epochs=4, batch_size=2048,
            mini_batch_size=256, total_timesteps=100000,
            save_interval=10000, log_interval=100, max_npcs=5,
            local_view_size=21, num_actions=6,
        )
        snapshot = build_config_snapshot(config, {"seed": 42, "custom_key": True})
        self.assertEqual(snapshot["seed"], 42)
        self.assertTrue(snapshot["custom_key"])


if __name__ == "__main__":
    unittest.main()
