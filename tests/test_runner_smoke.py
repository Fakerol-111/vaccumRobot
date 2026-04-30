"""Runner 编排层冒烟测试：core/trainer_runner.py + core/evaluator_runner.py

不启动完整训练/评估，只验证 orchestration 阶段不崩溃。
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from core.types import (
    TrainRequest,
    TrainResult,
    EvalRequest,
    EvalResult,
    RunContext,
    EvalContext,
    MapEvalResult,
)


class TestTrainRequestAssembly(unittest.TestCase):
    def test_minimal_train_request(self):
        req = TrainRequest(
            algo_config=SimpleNamespace(learning_rate=0.001),
            env_config={"default_npc_count": 1, "map_strategy": "round_robin"},
            curriculum={"enabled": False, "stages": []},
            training_config={"artifacts_dir": "artifacts"},
            general_config={"seed": 42},
            dashboard_config={"enabled": False, "host": "0.0.0.0", "port": 8088},
            metrics_config={"max_updates": 500, "max_episodes": 500},
            config_path=Path("/tmp/config.toml"),
            artifacts_root=Path("/tmp/artifacts"),
        )
        self.assertIsInstance(req, TrainRequest)
        self.assertEqual(req.general_config["seed"], 42)
        self.assertFalse(req.dashboard_config["enabled"])
        self.assertIsNone(req.run_id)
        self.assertIsNone(req.resume_from)

    def test_train_request_with_optional_fields(self):
        req = TrainRequest(
            algo_config=SimpleNamespace(learning_rate=0.001),
            env_config={},
            curriculum={"enabled": True, "stages": []},
            training_config={},
            general_config={},
            dashboard_config={},
            metrics_config={},
            config_path=Path("/tmp/config.toml"),
            artifacts_root=Path("/tmp/artifacts"),
            resume_from=Path("/tmp/checkpoint.pt"),
            run_id="test_run_001",
        )
        self.assertEqual(req.run_id, "test_run_001")
        self.assertEqual(req.resume_from, Path("/tmp/checkpoint.pt"))


class TestTrainResult(unittest.TestCase):
    def test_success_result(self):
        result = TrainResult(
            run_id="test_001",
            run_dir=Path("/tmp/run"),
            total_steps=10000,
        )
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_failure_result(self):
        result = TrainResult(
            run_id="test_001",
            run_dir=Path("/tmp/run"),
            total_steps=0,
            success=False,
            error="Something failed",
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Something failed")


class TestEvalRequestAssembly(unittest.TestCase):
    def test_minimal_eval_request(self):
        req = EvalRequest(
            map_ids=[1, 2],
            num_episodes=5,
            npc_count=1,
            station_count=4,
            run_id=None,
            step=None,
            gif_fps=10,
            output_dir=None,
            algo_config=SimpleNamespace(learning_rate=0.001),
            env_config={"default_npc_count": 1},
            artifacts_root=Path("/tmp/artifacts"),
        )
        self.assertIsInstance(req, EvalRequest)
        self.assertEqual(req.map_ids, [1, 2])
        self.assertIsNone(req.run_id)
        self.assertIsNone(req.output_dir)

    def test_eval_request_with_all_fields(self):
        req = EvalRequest(
            map_ids=[1],
            num_episodes=10,
            npc_count=2,
            station_count=3,
            run_id="test_run",
            step=5000,
            gif_fps=15,
            output_dir=Path("/tmp/eval_output"),
            algo_config=SimpleNamespace(),
            env_config={},
            artifacts_root=Path("/tmp/artifacts"),
        )
        self.assertEqual(req.run_id, "test_run")
        self.assertEqual(req.step, 5000)
        self.assertEqual(req.output_dir, Path("/tmp/eval_output"))


class TestEvalResult(unittest.TestCase):
    def test_default_result(self):
        result = EvalResult(
            checkpoints_root=Path("/tmp"),
            run_dir=Path("/tmp/run"),
            model_path=Path("/tmp/model.pt"),
            checkpoint_step=5000,
            eval_dir=Path("/tmp/eval"),
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.results), 0)
        self.assertEqual(result.total_episodes, 0)

    def test_with_results(self):
        map_result = MapEvalResult(
            map_name="map_1",
            rewards=[1.0, 2.0],
            steps=[100, 200],
            scores=[50, 60],
            charges=[3, 4],
        )
        result = EvalResult(
            checkpoints_root=Path("/tmp"),
            run_dir=Path("/tmp/run"),
            model_path=Path("/tmp/model.pt"),
            checkpoint_step=5000,
            eval_dir=Path("/tmp/eval"),
            results=[map_result],
            overall_avg_reward=1.5,
            overall_avg_score=55.0,
            overall_avg_steps=150.0,
            overall_avg_charges=3.5,
            total_episodes=2,
        )
        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.overall_avg_reward, 1.5)
        self.assertEqual(result.total_episodes, 2)
        self.assertEqual(result.results[0].avg_reward, 1.5)
        self.assertEqual(result.results[0].avg_score, 55.0)


class TestRunContext(unittest.TestCase):
    def test_run_context(self):
        ctx = RunContext(
            artifacts_root=Path("/tmp/artifacts"),
            checkpoints_root=Path("/tmp/checkpoints"),
            run_dir=Path("/tmp/run"),
            checkpoint_dir=Path("/tmp/run/checkpoints"),
            train_log_path=Path("/tmp/run/train.log"),
            run_id="test_001",
            seed=42,
        )
        self.assertEqual(ctx.run_id, "test_001")
        self.assertEqual(ctx.seed, 42)


class TestEvalContext(unittest.TestCase):
    def test_eval_context(self):
        ctx = EvalContext(
            checkpoints_root=Path("/tmp/checkpoints"),
            run_dir=Path("/tmp/run"),
            model_path=Path("/tmp/model.pt"),
            checkpoint_step=5000,
            eval_dir=Path("/tmp/eval"),
            map_configs=[{"map_id": 1}],
            map_names=["map_1"],
        )
        self.assertEqual(len(ctx.map_configs), 1)
        self.assertEqual(ctx.checkpoint_step, 5000)


class TestMapEvalResult(unittest.TestCase):
    def test_empty_stats(self):
        r = MapEvalResult(map_name="test")
        self.assertEqual(r.avg_reward, 0.0)
        self.assertEqual(r.avg_steps, 0.0)
        self.assertEqual(r.avg_score, 0.0)
        self.assertEqual(r.num_episodes, 0)

    def test_computed_stats(self):
        r = MapEvalResult(
            map_name="map_1",
            rewards=[1.0, 3.0],
            steps=[100, 200],
            scores=[50, 70],
            charges=[2, 4],
        )
        self.assertEqual(r.avg_reward, 2.0)
        self.assertEqual(r.avg_steps, 150.0)
        self.assertEqual(r.avg_score, 60.0)
        self.assertEqual(r.avg_charges, 3.0)
        self.assertEqual(r.num_episodes, 2)


if __name__ == "__main__":
    unittest.main()
