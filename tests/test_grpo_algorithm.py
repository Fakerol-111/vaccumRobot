"""GRPO algorithm tests.

Tests the algorithm-internal logic without requiring a live environment.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from agent.base import LossInfo
from agent.grpo.algorithm import GRPOAlgorithm, _sample_candidate_actions


# ── mock config ────────────────────────────────────────────

def _make_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        num_actions=8,
        learning_rate=0.0003,
        max_grad_norm=0.5,
        total_timesteps=10000,
        save_interval=5000,
        log_interval=100,
        max_npcs=5,
        local_view_size=21,
        # GRPO-specific
        branch_window=64,
        branch_interval=10,
        num_candidates=4,
        kl_coef=0.1,
        batch_size=256,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── _sample_candidate_actions tests ────────────────────────

class TestSampleCandidateActions(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1, 0.0, -1.0, -2.0]])
        self.legal = np.ones(8, dtype=np.float32)

    def test_returns_k_actions(self):
        actions = _sample_candidate_actions(self.logits, self.legal, k=3)
        self.assertEqual(len(actions), 3)
        for a in actions:
            self.assertIn(a, range(8))

    def test_respects_legal_mask(self):
        legal = np.array([0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
        actions = _sample_candidate_actions(self.logits, legal, k=3)
        # All returned actions must be legal
        for a in actions:
            self.assertEqual(legal[a], 1.0)

    def test_returns_empty_when_too_few_legal(self):
        legal = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        actions = _sample_candidate_actions(self.logits, legal, k=4)
        self.assertEqual(actions, [])

    def test_k_larger_than_num_actions(self):
        legal = np.ones(8, dtype=np.float32)
        actions = _sample_candidate_actions(self.logits, legal, k=20)
        self.assertLessEqual(len(actions), 8)
        self.assertGreater(len(actions), 0)

    def test_works_on_different_devices(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1, 0.0, -1.0, -2.0]],
                              device="cuda")
        actions = _sample_candidate_actions(logits, self.legal, k=3)
        self.assertEqual(len(actions), 3)


# ── GRPOAlgorithm lifecycle tests ─────────────────────────

class TestGRPOLifecycle(unittest.TestCase):
    def setUp(self):
        self.config = _make_config(branch_interval=10)
        self.algo = GRPOAlgorithm(self.config)

    def test_init_reference_synced(self):
        """Reference must be initialised as a copy of model."""
        for mp, rp in zip(self.algo.model.parameters(), self.algo.reference.parameters()):
            self.assertTrue(torch.equal(mp.data, rp.data))
        # Reference must be frozen
        for p in self.algo.reference.parameters():
            self.assertFalse(p.requires_grad)

    def test_init_steps_since_group_zero(self):
        self.assertEqual(self.algo._steps_since_group, 0)

    def test_on_step_increments_counter(self):
        dummy = (np.zeros((9, 21, 21)), np.zeros(10), np.ones(8))
        self.algo.on_step(*dummy, action=0, log_prob=0.0, value=0.0, reward=1.0, done=False)
        self.assertEqual(self.algo._steps_since_group, 1)

    def test_on_step_returns_none_before_interval(self):
        dummy = (np.zeros((9, 21, 21)), np.zeros(10), np.ones(8))
        for _ in range(9):
            result = self.algo.on_step(*dummy, action=0, log_prob=0.0, value=0.0, reward=1.0, done=False)
            self.assertIsNone(result)
        self.assertEqual(self.algo._steps_since_group, 9)

    def test_on_step_returns_lossinfo_at_interval(self):
        """At branch_interval, group_update is called. Without env_config it returns None."""
        dummy = (np.zeros((9, 21, 21)), np.zeros(10), np.ones(8))
        for _ in range(10):
            result = self.algo.on_step(*dummy, action=0, log_prob=0.0, value=0.0, reward=1.0, done=False)
        # Without env_config, group_update returns None
        self.assertIsNone(result)

    def test_on_step_counter_keeps_increasing_when_group_update_returns_none(self):
        """Counter is NOT reset when group_update returns None (no env_config)."""
        dummy = (np.zeros((9, 21, 21)), np.zeros(10), np.ones(8))
        for _ in range(10):
            self.algo.on_step(*dummy, action=0, log_prob=0.0, value=0.0, reward=1.0, done=False)
        self.assertEqual(self.algo._steps_since_group, 10)

    def test_set_branch_state_stores_state(self):
        state = {"env": "env_state", "pp": "pp_state"}
        self.algo.set_branch_state(state)
        self.assertEqual(self.algo._branch_state, state)

    def test_set_branch_state_none(self):
        self.algo.set_branch_state(None)
        self.assertIsNone(self.algo._branch_state)


# ── group_update early-exit tests ─────────────────────────

class TestGRPOGroupUpdateEarlyExit(unittest.TestCase):
    def setUp(self):
        self.config = _make_config(branch_interval=10, num_candidates=4)
        self.algo = GRPOAlgorithm(self.config)
        self.map_img = np.zeros((9, 21, 21), dtype=np.float32)
        self.vector = np.zeros(10, dtype=np.float32)
        self.legal_mask = np.ones(8, dtype=np.float32)

    def test_group_update_returns_none_when_env_config_none(self):
        result = self.algo.group_update(self.map_img, self.vector, self.legal_mask, env_config=None)
        self.assertIsNone(result)

    def test_group_update_returns_none_when_branch_state_none(self):
        self.algo._branch_state = None
        result = self.algo.group_update(
            self.map_img, self.vector, self.legal_mask,
            env_config={"map_id": 1},
        )
        self.assertIsNone(result)

    def test_group_update_returns_none_when_few_candidates(self):
        """If _sample_candidate_actions returns < 2 actions, group_update returns None."""
        self.algo.set_branch_state({"env": {}, "pp": {}})
        legal = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        result = self.algo.group_update(self.map_img, self.vector, legal, env_config={"map_id": 1})
        self.assertIsNone(result)

    def test_group_update_resets_counter_and_clears_state(self):
        """Successful group_update resets _steps_since_group to 0 and clears _branch_state."""
        self.algo.set_branch_state({"env": {"mock": True}, "pp": {"mock": True}})
        env_config = {"custom_map": [[1, 1], [1, 1]], "npc_count": 0, "station_count": 0}

        original_record = self.algo._record_npc_trace
        original_rollout = self.algo._rollout_branch
        self.algo._record_npc_trace = lambda *a, **kw: []
        self.algo._rollout_branch = lambda *a, **kw: 1.0

        # Also need to avoid the greedy argmax branch from requiring env
        original_get_logits = self.algo._get_logits
        self.algo._get_logits = lambda *a, **kw: torch.randn(1, 8)

        try:
            result = self.algo.group_update(self.map_img, self.vector, self.legal_mask, env_config)
        finally:
            self.algo._record_npc_trace = original_record
            self.algo._rollout_branch = original_rollout
            self.algo._get_logits = original_get_logits

        self.assertIsNotNone(result)
        self.assertIsInstance(result, LossInfo)
        self.assertEqual(self.algo._steps_since_group, 0)
        self.assertIsNone(self.algo._branch_state)


# ── metrics_reporter ──────────────────────────────────────

class TestGRPOMetricsReporter(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.algo = GRPOAlgorithm(self.config)

    def test_metrics_reporter_is_available(self):
        reporter = self.algo.metrics_reporter
        self.assertIsNotNone(reporter)

    def test_metrics_reporter_summary(self):
        reporter = self.algo.metrics_reporter
        summary = reporter.update_summary()
        self.assertIsInstance(summary, str)
        self.assertIn("mean_score", summary)

    def test_metrics_reporter_final_summary(self):
        reporter = self.algo.metrics_reporter
        lines = reporter.final_summary_lines()
        self.assertIsInstance(lines, list)
        self.assertTrue(len(lines) >= 1)

    def test_record_update(self):
        reporter = self.algo.metrics_reporter
        loss = LossInfo(
            total_loss=0.5, policy_loss=0.3, mean_reward=10.0,
            entropy=1.0,
            extra={"std_reward": 2.0, "kl_divergence": 0.01},
        )
        reporter.record_update(loss)
        summary = reporter.update_summary()
        self.assertIn("10.00", summary)
        self.assertIn("0.01", summary)


# ── ready_to_update / update ──────────────────────────────

class TestGRPOAlgoInterface(unittest.TestCase):
    def setUp(self):
        self.algo = GRPOAlgorithm(_make_config())

    def test_ready_to_update_returns_false(self):
        self.assertFalse(self.algo.ready_to_update())

    def test_update_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.algo.update()

    def test_compute_value_returns_zero(self):
        result = self.algo.compute_value(
            np.zeros((9, 21, 21)), np.zeros(10), np.ones(8),
        )
        self.assertEqual(result, 0.0)

    def test_set_env_config(self):
        env_config = {"map_id": 42}
        self.algo.set_env_config(env_config)
        self.assertEqual(self.algo._current_env_config, env_config)

    def test_exploit_returns_deterministic_action(self):
        map_img = np.zeros((9, 21, 21), dtype=np.float32)
        vector = np.zeros(10, dtype=np.float32)
        legal = np.ones(8, dtype=np.float32)
        r1 = self.algo.exploit(map_img, vector, legal)
        r2 = self.algo.exploit(map_img, vector, legal)
        self.assertEqual(r1.action, r2.action)
        self.assertIsNotNone(r1.log_prob)

    def test_explore_returns_action(self):
        map_img = np.zeros((9, 21, 21), dtype=np.float32)
        vector = np.zeros(10, dtype=np.float32)
        legal = np.ones(8, dtype=np.float32)
        result = self.algo.explore(map_img, vector, legal)
        self.assertIsInstance(result.action, int)
        self.assertIsNotNone(result.log_prob)


# ── checkpoint tests ──────────────────────────────────────

class TestGRPOCheckpoint(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.algo = GRPOAlgorithm(self.config)

    def test_save_checkpoint_includes_extra_state(self):
        import tempfile, os
        self.algo._steps_since_group = 7
        self.algo._episode_count = 3

        tmp = os.path.join(tempfile.gettempdir(), "grpo_ckpt_test.pt")
        self.algo.save_checkpoint(tmp, global_step=100, episode_counter=5)
        data = torch.load(tmp, map_location="cpu", weights_only=False)
        os.remove(tmp)

        self.assertIn("reference_state_dict", data)
        self.assertIn("steps_since_group", data)
        self.assertIn("episode_count", data)
        self.assertEqual(data["steps_since_group"], 7)
        self.assertEqual(data["episode_count"], 3)

    def test_load_checkpoint_restores_extra_state(self):
        import tempfile, os

        # Save with non-zero state
        self.algo._steps_since_group = 7
        self.algo._episode_count = 3
        tmp = os.path.join(tempfile.gettempdir(), "grpo_ckpt_load.pt")
        self.algo.save_checkpoint(tmp, global_step=100, episode_counter=5)

        # Create fresh algo and load
        algo2 = GRPOAlgorithm(self.config)
        self.assertEqual(algo2._steps_since_group, 0)
        self.assertEqual(algo2._episode_count, 0)

        algo2.load_checkpoint(tmp)
        os.remove(tmp)

        self.assertEqual(algo2._steps_since_group, 7)
        self.assertEqual(algo2._episode_count, 3)
        # Reference must still be frozen after loading
        for p in algo2.reference.parameters():
            self.assertFalse(p.requires_grad)


if __name__ == "__main__":
    unittest.main()
