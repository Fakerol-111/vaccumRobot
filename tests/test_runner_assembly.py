"""Runner assembly tests: algorithm instantiation, collector injection, set_env_config flow.

Tests the orchestration pattern in core/trainer_runner.py run_training().
Does not start full training — only verifies assembly phase.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import torch

from agent.base import Algorithm, LossInfo, MetricsReporter
from agent.registry import register, get


# ── mock algorithm for assembly testing ────────────────────

class _MockMetricsReporter(MetricsReporter):
    def __init__(self):
        super().__init__(collector=None)
        self.collector = None

    def record_update(self, loss_info: LossInfo) -> None:
        pass

    def update_summary(self) -> str:
        return "mock_algo summary"

    def final_summary_lines(self) -> list[str]:
        return ["mock final line"]

    def set_collector(self, collector: Any) -> None:
        self.collector = collector


class _AssemblyAlgo(Algorithm):
    def __init__(self, config, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")
        self._env_config = None
        self._metrics = _MockMetricsReporter()

    def explore(self, *a, **kw):
        from agent.base import ActResult
        return ActResult(action=0)

    def exploit(self, *a, **kw):
        from agent.base import ActResult
        return ActResult(action=1)

    def collect(self, *a, **kw): ...
    def ready_to_update(self): return False
    def update(self, *a, **kw): ...
    def compute_value(self, *a): return 0.0
    def save(self, *a): ...
    def load(self, *a): ...
    def save_checkpoint(self, *a, **kw): ...
    def load_checkpoint(self, *a, **kw): ...

    @property
    def metrics_reporter(self) -> _MockMetricsReporter:
        return self._metrics

    def set_env_config(self, env_config: dict[str, Any]) -> None:
        self._env_config = env_config


@register("assembly_test")
class _RegisteredAssemblyAlgo(_AssemblyAlgo):
    pass


# ── tests ─────────────────────────────────────────────────

class TestRunnerAssembly(unittest.TestCase):
    def test_get_algorithm_from_registry(self):
        cls = get("assembly_test")
        self.assertIs(cls, _RegisteredAssemblyAlgo)

    def test_algorithm_instantiation_with_config_and_device(self):
        cls = get("assembly_test")
        config = SimpleNamespace(learning_rate=0.001, num_actions=8)
        device = torch.device("cpu")
        algorithm = cls(config, device)
        self.assertIsInstance(algorithm, Algorithm)
        self.assertEqual(algorithm.config.learning_rate, 0.001)
        self.assertEqual(algorithm.device, device)

    def test_algorithm_instantiation_default_device(self):
        cls = get("assembly_test")
        config = SimpleNamespace(learning_rate=0.001, num_actions=8)
        algorithm = cls(config)
        self.assertEqual(algorithm.device.type, "cpu")

    def test_metrics_reporter_is_available(self):
        cls = get("assembly_test")
        algorithm = cls(SimpleNamespace(num_actions=8))
        reporter = algorithm.metrics_reporter
        self.assertIsNotNone(reporter)
        self.assertIsInstance(reporter, MetricsReporter)
        self.assertIsNone(reporter.collector)

    def test_collector_injection_flow(self):
        """Simulate: reporter = algorithm.metrics_reporter; reporter.set_collector(collector)"""
        cls = get("assembly_test")
        algorithm = cls(SimpleNamespace(num_actions=8))
        reporter = algorithm.metrics_reporter
        mock_collector = object()
        reporter.set_collector(mock_collector)
        self.assertIs(reporter.collector, mock_collector)

    def test_set_env_config_flow(self):
        """Simulate: algorithm.set_env_config(env_config) called by Trainer."""
        cls = get("assembly_test")
        algorithm = cls(SimpleNamespace(num_actions=8))
        env_config = {"map_id": 1, "npc_count": 3}
        algorithm.set_env_config(env_config)
        self.assertEqual(algorithm._env_config, env_config)

    def test_full_assembly_pipeline(self):
        """Simulate the full assembly from core/trainer_runner.py run_training()."""
        # 1. Look up class
        algo_cls = get("assembly_test")

        # 2. Instantiate with config + device
        algo_config = SimpleNamespace(
            learning_rate=0.0003, num_actions=8, max_grad_norm=0.5,
            total_timesteps=1000, save_interval=500, log_interval=100,
            max_npcs=5, local_view_size=21, gamma=0.99,
        )
        device = torch.device("cpu")
        algorithm = algo_cls(algo_config, device)

        # 3. Inject collector
        mock_collector = SimpleNamespace(add_event=lambda *a: None)
        reporter = algorithm.metrics_reporter
        if reporter is not None:
            reporter.set_collector(mock_collector)
        self.assertIs(reporter.collector, mock_collector)

        # 4. Set env config
        env_config = {
            "default_npc_count": 1, "default_station_count": 4,
            "map_strategy": "round_robin", "default_map_list": [1],
        }
        algorithm.set_env_config(env_config)
        self.assertEqual(algorithm._env_config, env_config)

        # 5. Verify all properties survive
        self.assertEqual(algorithm.config.learning_rate, 0.0003)
        self.assertEqual(algorithm.device, device)


if __name__ == "__main__":
    unittest.main()
