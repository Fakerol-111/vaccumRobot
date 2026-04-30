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


if __name__ == "__main__":
    unittest.main()
