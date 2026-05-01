"""Config dispatch tests: scripts/train.py dispatch logic.

Tests the algorithm config dispatch pattern:
  - algo_name validation against registry
  - getattr-based config section selection
  - fallback when algo section is missing from bundle
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.registry import register, list_available


class _DummyAlgo:
    """Not a real Algorithm subclass — only needed to populate registry."""

    pass


class TestConfigDispatch(unittest.TestCase):
    """Test the dispatch pattern used in scripts/train.py main():

    algo_name = cfg.algo["name"]
    if algo_name not in list_available():
        raise ValueError(...)
    algo_config = getattr(cfg, algo_name, None)
    """

    @classmethod
    def setUpClass(cls):
        # Ensure registry has test entries
        register("ppo")(_DummyAlgo)
        register("grpo")(_DummyAlgo)

    def test_algo_name_in_available(self):
        names = list_available()
        self.assertIn("ppo", names)
        self.assertIn("grpo", names)

    def test_validation_passes_for_known_algo(self):
        algo_name = "ppo"
        self.assertIn(algo_name, list_available())

    def test_validation_fails_for_unknown_algo(self):
        algo_name = "unknown"
        self.assertNotIn(algo_name, list_available())

    def test_getattr_selects_ppo_section(self):
        cfg = SimpleNamespace(
            algo={"name": "ppo"},
            ppo=SimpleNamespace(learning_rate=0.0003),
            grpo=SimpleNamespace(learning_rate=0.0001),
        )
        algo_name = cfg.algo["name"]
        algo_config = getattr(cfg, algo_name, None)
        self.assertIsNotNone(algo_config)
        self.assertEqual(algo_config.learning_rate, 0.0003)

    def test_getattr_selects_grpo_section(self):
        cfg = SimpleNamespace(
            algo={"name": "grpo"},
            ppo=SimpleNamespace(learning_rate=0.0003),
            grpo=SimpleNamespace(learning_rate=0.0001),
        )
        algo_name = cfg.algo["name"]
        algo_config = getattr(cfg, algo_name, None)
        self.assertIsNotNone(algo_config)
        self.assertEqual(algo_config.learning_rate, 0.0001)

    def test_getattr_fallback_none_when_section_missing(self):
        cfg = SimpleNamespace(
            algo={"name": "grpo"},
            ppo=SimpleNamespace(learning_rate=0.0003),
            # no grpo section
        )
        algo_name = cfg.algo["name"]
        algo_config = getattr(cfg, algo_name, None)
        self.assertIsNone(algo_config)

    def test_algo_name_from_config_bundle(self):
        """Simulate load_train_config_bundle().algo["name"]."""
        cfg = SimpleNamespace(algo={"name": "grpo"})
        self.assertEqual(cfg.algo["name"], "grpo")
        self.assertIn(cfg.algo["name"], list_available())

    def test_complete_dispatch_flow(self):
        """Simulate the full dispatch as done in train.py main()."""
        cfg = SimpleNamespace(
            algo={"name": "ppo"},
            ppo=SimpleNamespace(batch_size=512),
            grpo=SimpleNamespace(batch_size=256),
        )
        algo_name = cfg.algo["name"]
        self.assertIn(algo_name, list_available())
        algo_config = getattr(cfg, algo_name, None)
        self.assertIsNotNone(algo_config)
        self.assertEqual(algo_config.batch_size, 512)


class TestEvalDispatchChain(unittest.TestCase):
    """Test the eval dispatch chain: algo_name → config → registry lookup.

    This mirrors the pattern in scripts/eval.py and core/evaluator_runner.py.
    """

    @classmethod
    def setUpClass(cls):
        from agent.registry import register as _register
        from agent.base import Algorithm

        class _DummyEvalAlgo(Algorithm):
            def __init__(self, config=None, device=None):
                self.config = config
                self.device = device

            def explore(self, *args, **kwargs): ...
            def exploit(self, *args, **kwargs): ...
            def collect(self, *args, **kwargs): ...
            def on_step(self, *args, **kwargs): ...
            def ready_to_update(self):
                return False

            def update(self, **kwargs): ...
            def compute_value(self, *args):
                return 0.0

            def save(self, path): ...
            def load(self, path): ...
            @property
            def metrics_reporter(self): ...

            def load_checkpoint(self, path): ...
            def save_checkpoint(self, path, **kwargs): ...

        cls.algo_cls = _register("eval_test_dummy")(_DummyEvalAlgo)

    def test_registry_get_by_name(self):
        from agent.registry import get as get_algo

        cls = get_algo("eval_test_dummy")
        self.assertIs(cls, self.algo_cls)

    def test_unknown_algo_raises(self):
        from agent.registry import get as get_algo

        with self.assertRaises(ValueError):
            get_algo("nonexistent_algo")

    def test_eval_flow_simulated(self):
        """Simulate the full eval.py + evaluator_runner.py flow."""
        from agent.registry import get as get_algo
        from types import SimpleNamespace

        train_cfg = SimpleNamespace(
            algo={"name": "eval_test_dummy"},
            eval_test_dummy=SimpleNamespace(num_actions=8, learning_rate=0.001),
        )

        # Step 1: resolve algo name and config (eval.py)
        algo_name = train_cfg.algo["name"]
        algo_config = getattr(train_cfg, algo_name, None)
        self.assertIsNotNone(algo_config)

        # Step 2: registry lookup + instantiation (evaluator_runner.py)
        algo_cls = get_algo(algo_name)
        algo = algo_cls(algo_config, device="cpu")
        self.assertIsInstance(algo, self.algo_cls)


if __name__ == "__main__":
    unittest.main()
