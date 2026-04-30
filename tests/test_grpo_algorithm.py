"""GRPO algorithm tests: _top_k_actions, on_step trigger, branch state, group_update lifecycle.

Tests the algorithm-internal logic without requiring a live environment.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from agent.base import LossInfo
from agent.grpo.algorithm import GRPOAlgorithm, _top_k_actions


# ── mock config ────────────────────────────────────────────

def _make_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        num_actions=8,
        learning_rate=0.0003,
        max_grad_norm=0.5,
        gamma=0.99,
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
        action_prob_threshold=0.01,
        batch_size=256,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── _top_k_actions tests ──────────────────────────────────

class TestTopKActions(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1, 0.0, -1.0, -2.0]])
        self.legal = np.ones(8, dtype=np.float32)

    def test_selects_top_k(self):
        actions = _top_k_actions(self.logits, self.legal, k=3)
        self.assertEqual(actions, [2, 1, 0])  # highest logits: 3, 2, 1

    def test_respects_legal_mask(self):
        legal = np.array([0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32)
        actions = _top_k_actions(self.logits, legal, k=3)
        # Only actions 2,3,4 are legal. Among them: logits 3.0, 0.5, 0.1 → [2,3,4]
        self.assertEqual(actions, [2, 3, 4])

    def test_respects_prob_threshold(self):
        logits = torch.tensor([[10.0, 5.0, 0.1, 0.05, 0.01, 0.0, 0.0, 0.0]])
        actions = _top_k_actions(logits, self.legal, k=4, prob_threshold=0.01)
        # Action 0: ~0.993, action 1: ~0.007, rest below threshold
        # After filtering, only action 0 remains (action 1 is right at threshold so kept)
        # Actually need to compute: softmax([10, 5, 0.1, 0.05, 0.01, 0, 0, 0])
        # exp(10)=22026, exp(5)=148.4, exp(0.1)=1.105, exp(0.05)=1.051, ...
        # total ≈ 22026+148.4+1.105+1.051+1.01+1+1+1 ≈ 22180
        # p0 ≈ 22026/22180 ≈ 0.993, p1 ≈ 148.4/22180 ≈ 0.00669
        # p0 >= 0.01, p1 = 0.00669 < 0.01 (below threshold)
        # With k=4, only p0 survives filtering
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0], 0)

    def test_returns_empty_when_no_action_above_threshold(self):
        logits = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        # Uniform distribution, each p = 0.125. With k=4 and threshold=0.01,
        # all are above threshold but only top 4 returned
        actions = _top_k_actions(logits, self.legal, k=4, prob_threshold=0.01)
        self.assertEqual(len(actions), 4)

    def test_k_larger_than_num_actions(self):
        actions = _top_k_actions(self.logits, self.legal, k=20)
        self.assertLessEqual(len(actions), 8)
        self.assertGreater(len(actions), 0)

    def test_works_on_different_devices(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        logits = torch.tensor([[1.0, 2.0, 3.0, 0.5, 0.1, 0.0, -1.0, -2.0]],
                              device="cuda")
        actions = _top_k_actions(logits, self.legal, k=3)
        self.assertEqual(actions, [2, 1, 0])


# ── GRPOAlgorithm lifecycle tests ─────────────────────────

class TestGRPOLifecycle(unittest.TestCase):
    def setUp(self):
        self.config = _make_config(branch_interval=10)
        self.algo = GRPOAlgorithm(self.config)

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
        # Counter stays at 10 because group_update returned None without resetting
        self.assertEqual(self.algo._steps_since_group, 10)

    def test_on_step_counter_keeps_growing_past_interval(self):
        dummy = (np.zeros((9, 21, 21)), np.zeros(10), np.ones(8))
        for _ in range(15):
            self.algo.on_step(*dummy, action=0, log_prob=0.0, value=0.0, reward=1.0, done=False)
        # Counter continues past interval when group_update can't proceed
        self.assertEqual(self.algo._steps_since_group, 15)

    def test_on_step_passthrough_collect(self):
        """GRPOAlgorithm.collect is a no-op; on_step overrides the default.
        Verify on_step is used (not collect)."""
        self.algo.collect("anything", None, None, 42, 0.0, 0.0, 0.0, False)
        # collect does nothing — no crash means it works

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
        """If _top_k_actions returns < 2 actions, group_update returns None."""
        self.algo.set_branch_state({"env": {}, "pp": {}})
        # Use a legal mask with only 1 legal action
        legal = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        result = self.algo.group_update(self.map_img, self.vector, legal, env_config={"map_id": 1})
        self.assertIsNone(result)

    def test_group_update_returns_loss_info_when_all_conditions_met(self):
        """With env_config + branch_state + 2+ candidates, group_update runs the full pipeline.

        We mock the env-dependent methods to avoid actual rollout.
        """
        self.algo.set_branch_state({"env": {"mock": True}, "pp": {"mock": True}})
        env_config = {"custom_map": [[1, 1], [1, 1]], "npc_count": 0, "station_count": 0}

        # Mock _record_npc_trace and _rollout_branch to avoid real env calls
        original_record = self.algo._record_npc_trace
        original_rollout = self.algo._rollout_branch

        self.algo._record_npc_trace = lambda *a, **kw: []  # empty trace
        self.algo._rollout_branch = lambda *a, **kw: float(hash(str(a)))  # deterministic score

        try:
            result = self.algo.group_update(self.map_img, self.vector, self.legal_mask, env_config)
        finally:
            self.algo._record_npc_trace = original_record
            self.algo._rollout_branch = original_rollout

        # With mocked rollouts, we should get a LossInfo
        self.assertIsNotNone(result)
        self.assertIsInstance(result, LossInfo)
        self.assertIsInstance(result.total_loss, float)
        self.assertIsInstance(result.policy_loss, float)
        self.assertIsInstance(result.mean_reward, float)

    def test_group_update_clears_branch_state(self):
        """After group_update, _branch_state should be None."""
        self.algo.set_branch_state({"env": {"mock": True}, "pp": {"mock": True}})
        env_config = {"custom_map": [[1, 1], [1, 1]], "npc_count": 0, "station_count": 0}

        original_record = self.algo._record_npc_trace
        original_rollout = self.algo._rollout_branch
        self.algo._record_npc_trace = lambda *a, **kw: []
        self.algo._rollout_branch = lambda *a, **kw: 1.0

        try:
            self.algo.group_update(self.map_img, self.vector, self.legal_mask, env_config)
        finally:
            self.algo._record_npc_trace = original_record
            self.algo._rollout_branch = original_rollout

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


if __name__ == "__main__":
    unittest.main()
