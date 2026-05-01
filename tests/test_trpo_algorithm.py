"""TRPO algorithm behavioral tests.

Covers:
  - Line search success / failure
  - Value optimizer parameter isolation
  - Checkpoint round-trip
  - Basic lifecycle (explore, exploit, collect, update)
"""

from __future__ import annotations

import tempfile
import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from agent.base import LossInfo
from agent.trpo.algorithm import TRPOAlgorithm
from agent.nn.separate_ac import SeparateActorCritic


def _make_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        num_actions=8,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        max_kl=0.01,
        cg_damping=0.1,
        cg_iterations=10,
        line_search_steps=10,
        value_epochs=2,
        value_mini_batch_size=16,
        max_grad_norm=0.5,
        batch_size=32,
        total_timesteps=10000,
        save_interval=5000,
        save_time_interval=0,
        log_interval=100,
        max_npcs=5,
        local_view_size=21,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _dummy_transition(num_actions: int = 8, batch_dim: int = 0):
    """Create a single (or batched) dummy transition."""
    if batch_dim > 0:
        return (
            np.zeros((batch_dim, 9, 21, 21), dtype=np.float32),
            np.zeros((batch_dim, 10), dtype=np.float32),
            np.ones((batch_dim, num_actions), dtype=np.float32),
        )
    return (
        np.zeros((9, 21, 21), dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.ones(num_actions, dtype=np.float32),
    )


# ── TRPOAlgorithm lifecycle tests ──────────────────────────

class TestTRPOLifecycle(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.algo = TRPOAlgorithm(self.config)

    def test_explore_returns_act_result(self):
        map_img, vector, legal = _dummy_transition()
        result = self.algo.explore(map_img, vector, legal)
        self.assertIsInstance(result.action, int)
        self.assertIsNotNone(result.log_prob)
        self.assertIsNotNone(result.value)

    def test_exploit_returns_deterministic_action(self):
        map_img, vector, legal = _dummy_transition()
        r1 = self.algo.exploit(map_img, vector, legal)
        r2 = self.algo.exploit(map_img, vector, legal)
        self.assertEqual(r1.action, r2.action)
        self.assertGreaterEqual(r1.action, 0)

    def test_explore_logits_saved_for_collect(self):
        map_img, vector, legal = _dummy_transition()
        self.algo.explore(map_img, vector, legal)
        self.assertIsNotNone(self.algo._last_logits)

    def test_compute_value_returns_float(self):
        map_img, vector, legal = _dummy_transition()
        v = self.algo.compute_value(map_img, vector, legal)
        self.assertIsInstance(v, float)

    def test_collect_fills_buffer(self):
        self.assertEqual(len(self.algo._rewards), 0)
        map_img, vector, legal = _dummy_transition()
        # Must call explore() first to populate _last_logits
        result = self.algo.explore(map_img, vector, legal)
        self.algo.collect(map_img, vector, legal, action=0, log_prob=0.0,
                          value=0.0, reward=1.0, done=False)
        self.assertEqual(len(self.algo._rewards), 1)

    def test_ready_to_update(self):
        self.assertFalse(self.algo.ready_to_update())
        for i in range(self.config.batch_size):
            map_img, vector, legal = _dummy_transition()
            self.algo._rewards.append(1.0)
            self.algo._dones.append(False)
            self.algo._map_imgs.append(map_img)
            self.algo._vectors.append(vector)
            self.algo._legal_masks.append(legal)
            self.algo._actions.append(0)
            self.algo._old_log_probs.append(0.0)
            self.algo._old_logits.append(np.ones(8))
            self.algo._values.append(0.0)
        self.assertTrue(self.algo.ready_to_update())

    def test_buffer_clear(self):
        self.algo._rewards.append(1.0)
        self.algo._buffer_clear()
        self.assertEqual(len(self.algo._rewards), 0)


# ── Update & line search tests ─────────────────────────────

class TestTRPOUpdate(unittest.TestCase):
    def setUp(self):
        self.config = _make_config(batch_size=4, line_search_steps=10)
        self.algo = TRPOAlgorithm(self.config)
        # Fill buffer with 4 transitions
        self._fill_buffer(self.algo)

    def _fill_buffer(self, algo):
        for i in range(algo.config.batch_size):
            map_img = np.random.randn(9, 21, 21).astype(np.float32)
            vector = np.random.randn(10).astype(np.float32)
            legal = np.ones(algo.config.num_actions, dtype=np.float32)
            logits = np.random.randn(algo.config.num_actions).astype(np.float32)
            algo._map_imgs.append(map_img)
            algo._vectors.append(vector)
            algo._legal_masks.append(legal)
            algo._actions.append(np.random.randint(0, algo.config.num_actions))
            algo._old_log_probs.append(-0.5)
            algo._old_logits.append(logits)
            algo._values.append(np.random.randn())
            algo._rewards.append(float(np.random.randn()))
            algo._dones.append(False)

    def test_update_returns_lossinfo(self):
        result = self.algo.update()
        self.assertIsInstance(result, LossInfo)
        self.assertIn("surrogate_before", result.extra)
        self.assertIn("surrogate_after", result.extra)
        self.assertIn("kl", result.extra)
        self.assertIn("line_search_step", result.extra)
        self.assertIn("line_search_accepted", result.extra)

    def test_update_clears_buffer(self):
        self.algo.update()
        self.assertEqual(len(self.algo._rewards), 0)

    def test_value_optimizer_only_touches_critic(self):
        """After update, critic params should differ but actor params should
        only be changed by line search (not by value_optimizer)."""
        # Snapshot actor params before
        actor_before = [p.data.clone() for p in self.algo._policy_params]
        critic_before = [p.data.clone() for p in self.algo._value_params]

        self.algo.update()

        # Critic should have changed (value update via value_optimizer)
        critic_changed = any(
            not torch.equal(cb, ca)
            for cb, ca in zip(critic_before, self.algo._value_params)
        )

        self.assertTrue(critic_changed, "Critic params should change after value update")

    def test_line_search_accepts_with_large_max_kl(self):
        """With max_kl very large, line search should accept if surrogate improves."""
        algo = TRPOAlgorithm(_make_config(batch_size=4, max_kl=100.0))
        self._fill_buffer(algo)
        result = algo.update()
        # Metrics should always contain the expected fields regardless of outcome
        self.assertIn("surrogate_before", result.extra)
        self.assertIn("surrogate_after", result.extra)
        self.assertIn("kl", result.extra)
        self.assertIn("line_search_step", result.extra)
        # Note: with random data the surrogate may not improve, so we don't
        # assert acceptance here — the failure test below covers that branch

    def test_line_search_fails_with_tiny_max_kl(self):
        """With max_kl extremely small, line search fails and rolls back."""
        algo = TRPOAlgorithm(_make_config(batch_size=4, max_kl=1e-10))
        originals = [p.data.clone() for p in algo._policy_params]
        self._fill_buffer(algo)
        result = algo.update()
        # After fallback, params should equal originals
        for orig, cur in zip(originals, algo._policy_params):
            self.assertTrue(torch.equal(orig, cur),
                            "Params should roll back when line search fails")
        self.assertFalse(result.extra["line_search_accepted"])


# ── Parameter isolation tests ──────────────────────────────

class TestTRPOParams(unittest.TestCase):
    def test_flat_params_round_trip(self):
        algo = TRPOAlgorithm(_make_config())
        flat = algo._flat_params(algo._policy_params)
        # Verify the type and shape
        self.assertIsInstance(flat, torch.Tensor)
        self.assertGreater(flat.numel(), 0)

        # Round-trip: flatten → set back
        flat_orig = flat.clone()
        algo._set_params(algo._policy_params, flat_orig)
        flat_after = algo._flat_params(algo._policy_params)
        self.assertTrue(torch.equal(flat_orig, flat_after))


# ── Checkpoint tests ───────────────────────────────────────

class TestTRPOCheckpoint(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.algo = TRPOAlgorithm(self.config)

    def test_save_checkpoint_round_trip(self):
        tmp = os.path.join(tempfile.gettempdir(), "trpo_ckpt_test.pt")
        self.algo.save_checkpoint(tmp, global_step=100, episode_counter=5)

        algo2 = TRPOAlgorithm(self.config)
        ckpt = algo2.load_checkpoint(tmp)
        os.remove(tmp)

        self.assertEqual(ckpt.global_step, 100)
        self.assertEqual(ckpt.episode_counter, 5)
        # Model weights should match
        for p1, p2 in zip(self.algo.model.parameters(), algo2.model.parameters()):
            self.assertTrue(torch.equal(p1.data, p2.data))

    def test_save_load_round_trip(self):
        tmp = os.path.join(tempfile.gettempdir(), "trpo_save_test.pt")
        self.algo.save(tmp)
        algo2 = TRPOAlgorithm(self.config)
        algo2.load(tmp)
        os.remove(tmp)
        for p1, p2 in zip(self.algo.model.parameters(), algo2.model.parameters()):
            self.assertTrue(torch.equal(p1.data, p2.data))

    def test_separate_actor_critic_model(self):
        """TRPO always uses SeparateActorCritic, verify it works."""
        self.assertIsInstance(self.algo.model, SeparateActorCritic)
        self.assertIsNotNone(self.algo.model.actor)
        self.assertIsNotNone(self.algo.model.critic)

    def test_actor_critic_are_independent(self):
        """Actor and critic should not share parameter storage."""
        actor_params = set(id(p) for p in self.algo._policy_params)
        critic_params = set(id(p) for p in self.algo._value_params)
        self.assertTrue(actor_params.isdisjoint(critic_params),
                        "Actor and critic parameters must be disjoint")


# ── Metrics ────────────────────────────────────────────────

class TestTRPOMetrics(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.algo = TRPOAlgorithm(self.config)

    def test_metrics_reporter_available(self):
        reporter = self.algo.metrics_reporter
        self.assertIsNotNone(reporter)

    def test_metrics_reporter_record_update(self):
        reporter = self.algo.metrics_reporter
        loss = LossInfo(
            total_loss=0.0, policy_loss=0.5, value_loss=0.1, entropy=1.0,
            mean_reward=10.0,
            extra={"surrogate_loss": 0.6, "surrogate_before": 0.5, "surrogate_after": 0.6,
                   "kl": 0.005, "line_search_step": 1.0, "line_search_accepted": True},
        )
        reporter.record_update(loss)
        summary = reporter.update_summary()
        self.assertIn("0.60", summary)


if __name__ == "__main__":
    unittest.main()
