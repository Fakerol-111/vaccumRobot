"""Registry framework tests: agent/registry.py

Tests: register, get, list_available, duplicate registration, unknown algorithm error.
"""

from __future__ import annotations

import unittest

from agent.base import Algorithm
from agent.registry import register, get, list_available


# ── helpers ────────────────────────────────────────────────

class _DummyAlgo(Algorithm):
    def explore(self, *a, **kw): ...
    def exploit(self, *a, **kw): ...
    def collect(self, *a, **kw): ...
    def ready_to_update(self): return False
    def update(self, *a, **kw): ...
    def compute_value(self, *a): return 0.0
    def save(self, *a): ...
    def load(self, *a): ...
    def save_checkpoint(self, *a, **kw): ...
    def load_checkpoint(self, *a, **kw): ...


# ── tests ─────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):
    def setUp(self):
        # Fresh module-level registry for test isolation.
        # We import the internal _registry and clear + restore it.
        import agent.registry as regmod
        self._orig_registry = regmod._registry.copy()
        regmod._registry.clear()

    def tearDown(self):
        import agent.registry as regmod
        regmod._registry.clear()
        regmod._registry.update(self._orig_registry)

    def test_register_and_get(self):
        register("test_algo")(_DummyAlgo)
        cls = get("test_algo")
        self.assertIs(cls, _DummyAlgo)

    def test_get_returns_same_class(self):
        register("my_algo")(_DummyAlgo)
        algo = get("my_algo")
        self.assertTrue(issubclass(algo, Algorithm))

    def test_list_available_empty(self):
        self.assertEqual(list_available(), [])

    def test_list_available_sorted(self):
        register("z_algo")(_DummyAlgo)
        register("a_algo")(_DummyAlgo)
        self.assertEqual(list_available(), ["a_algo", "z_algo"])

    def test_duplicate_registration_overwrites(self):
        class First(_DummyAlgo):
            pass
        class Second(_DummyAlgo):
            pass

        register("dup")(First)
        self.assertIs(get("dup"), First)

        register("dup")(Second)
        self.assertIs(get("dup"), Second)

    def test_unknown_algorithm_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))
        self.assertIn("Available", str(ctx.exception))

    def test_register_decorator_preserves_class(self):
        @register("decorated")
        class DecoratedAlgo(_DummyAlgo):
            pass

        self.assertIs(get("decorated"), DecoratedAlgo)
        self.assertIn("decorated", list_available())


if __name__ == "__main__":
    unittest.main()
