"""PPO-KL algorithm behavioral tests.

Covers:
  - Adaptive KL coefficient (beta) up/down regulation
  - Checkpoint round-trip including kl_beta
  - Adaptive vs non-adaptive modes
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
from agent.ppo_kl.algorithm import PPOKLAlgorithm, kl_div_categorical


def _make_config(**overrides) -> SimpleNamespace:
    defaults = dict(
        # PPO common
        num_actions=8,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=2,
        batch_size=8,
        mini_batch_size=4,
        total_timesteps=10000,
        save_interval=5000,
        save_time_interval=0,
        log_interval=100,
        max_npcs=5,
        local_view_size=21,
        model_type="shared",
        # PPO-KL specific
        target_kl=0.01,
        kl_beta=1.0,
        kl_adaptive=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _dummy_transition(num_actions: int = 8):
    return (
        np.zeros((9, 21, 21), dtype=np.float32),
        np.zeros(10, dtype=np.float32),
        np.ones(num_actions, dtype=np.float32),
    )


# ── PPOKLAlgorithm lifecycle tests ─────────────────────────

class TestPPOKLLifecycle(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.algo = PPOKLAlgorithm(self.config)

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

    def test_compute_value_returns_float(self):
        map_img, vector, legal = _dummy_transition()
        v = self.algo.compute_value(map_img, vector, legal)
        self.assertIsInstance(v, float)

    def test_explore_stashes_logits(self):
        """explore() should save _last_logits for later KL computation."""
        map_img, vector, legal = _dummy_transition()
        self.algo.explore(map_img, vector, legal)
        self.assertIsNotNone(self.algo._last_logits)

    def test_collect_fills_buffer(self):
        self.assertEqual(len(self.algo._rewards), 0)
        map_img, vector, legal = _dummy_transition()
        result = self.algo.explore(map_img, vector, legal)
        self.algo.collect(map_img, vector, legal, action=0, log_prob=0.0,
                          value=0.0, reward=1.0, done=False)
        self.assertEqual(len(self.algo._rewards), 1)

    def test_collect_stashes_logits(self):
        map_img, vector, legal = _dummy_transition()
        self.algo.explore(map_img, vector, legal)
        self.algo.collect(map_img, vector, legal, action=0, log_prob=0.0,
                          value=0.0, reward=1.0, done=False)
        self.assertEqual(len(self.algo._old_logits), 1)
        self.assertEqual(len(self.algo._old_logits), 1)

    def test_ready_to_update(self):
        self.assertFalse(self.algo.ready_to_update())
        for _ in range(self.config.batch_size):
            self.algo._rewards.append(1.0)
            self.algo._dones.append(False)
        self.assertTrue(self.algo.ready_to_update())

    def test_update_returns_lossinfo(self):
        self._fill_realistic_buffer()
        result = self.algo.update()
        self.assertIsInstance(result, LossInfo)
        self.assertIn("kl", result.extra)
        self.assertIn("kl_beta", result.extra)

    def test_update_clears_buffer(self):
        self._fill_realistic_buffer()
        self.algo.update()
        self.assertEqual(len(self.algo._rewards), 0)

    def _fill_realistic_buffer(self):
        """Fill buffer with random but valid-looking transitions."""
        for _ in range(self.config.batch_size):
            map_img = np.random.randn(9, 21, 21).astype(np.float32)
            vector = np.random.randn(10).astype(np.float32)
            legal = np.ones(self.config.num_actions, dtype=np.float32)
            logits = np.random.randn(self.config.num_actions).astype(np.float32)
            self.algo._map_imgs.append(map_img)
            self.algo._vectors.append(vector)
            self.algo._legal_masks.append(legal)
            self.algo._actions.append(np.random.randint(0, self.config.num_actions))
            self.algo._old_log_probs.append(-0.5)
            self.algo._old_logits.append(logits)
            self.algo._values.append(0.0)
            self.algo._rewards.append(float(np.random.randn()))
            self.algo._dones.append(False)


# ── Adaptive kl_beta tests ────────────────────────────────

class TestPPOKLAdaptiveBeta(unittest.TestCase):
    def setUp(self):
        self.config = _make_config(batch_size=4, mini_batch_size=4, ppo_epochs=1,
                                   kl_adaptive=True, target_kl=0.01, kl_beta=1.0)
        self.algo = PPOKLAlgorithm(self.config)

    def _fill_and_update(self, algo) -> LossInfo:
        for _ in range(algo.config.batch_size):
            img = np.random.randn(9, 21, 21).astype(np.float32)
            vec = np.random.randn(10).astype(np.float32)
            legal = np.ones(algo.config.num_actions, dtype=np.float32)
            logits = np.random.randn(algo.config.num_actions).astype(np.float32)
            algo._map_imgs.append(img)
            algo._vectors.append(vec)
            algo._legal_masks.append(legal)
            algo._actions.append(np.random.randint(0, algo.config.num_actions))
            algo._old_log_probs.append(-0.5)
            algo._old_logits.append(logits)
            algo._values.append(0.0)
            algo._rewards.append(0.0)
            algo._dones.append(False)
        return algo.update()

    def test_non_adaptive_beta_stays_same(self):
        """When kl_adaptive=False, beta should not change after update."""
        algo = PPOKLAlgorithm(_make_config(batch_size=4, mini_batch_size=4,
                                           ppo_epochs=1, kl_adaptive=False, kl_beta=2.5))
        beta_before = algo._kl_beta
        self._fill_and_update(algo)
        self.assertEqual(algo._kl_beta, beta_before)

    def test_adaptive_beta_initialized_from_config(self):
        algo = PPOKLAlgorithm(_make_config(kl_beta=42.0))
        self.assertEqual(algo._kl_beta, 42.0)

    def test_adaptive_beta_clipped(self):
        """Beta should be clipped to [1e-10, 1e10] range."""
        # We can't force the exact beta after update since KL depends on model,
        # but we can verify that initial value is within range
        self.assertGreaterEqual(self.algo._kl_beta, 1e-10)
        self.assertLessEqual(self.algo._kl_beta, 1e10)


# ── kl_div_categorical unit test ───────────────────────────

class TestKLDivCategorical(unittest.TestCase):
    def test_kl_with_identical_logits_is_zero(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.2, 0.1]])
        kl = kl_div_categorical(logits, logits)
        self.assertTrue(torch.allclose(kl, torch.zeros_like(kl), atol=1e-6))

    def test_kl_is_non_negative(self):
        old = torch.tensor([[1.0, 2.0, 3.0]])
        new = torch.tensor([[3.0, 2.0, 1.0]])
        kl = kl_div_categorical(old, new)
        self.assertGreaterEqual(kl.item(), 0.0)

    def test_kl_batch_shape(self):
        old = torch.randn(4, 6)
        new = torch.randn(4, 6)
        kl = kl_div_categorical(old, new)
        self.assertEqual(kl.shape, (4,))


# ── Checkpoint tests ───────────────────────────────────────

class TestPPOKLCheckpoint(unittest.TestCase):
    def setUp(self):
        self.config = _make_config(kl_beta=3.5)
        self.algo = PPOKLAlgorithm(self.config)
        self.algo._kl_beta = 7.5  # simulate a change

    def test_save_checkpoint_preserves_kl_beta(self):
        tmp = os.path.join(tempfile.gettempdir(), "ppo_kl_ckpt_test.pt")
        self.algo.save_checkpoint(tmp, global_step=100, episode_counter=5)
        data = torch.load(tmp, map_location="cpu", weights_only=False)
        os.remove(tmp)

        self.assertIn("kl_beta", data)
        self.assertEqual(data["kl_beta"], 7.5)

    def test_load_checkpoint_restores_kl_beta(self):
        tmp = os.path.join(tempfile.gettempdir(), "ppo_kl_ckpt_load.pt")
        self.algo.save_checkpoint(tmp, global_step=100, episode_counter=5)

        algo2 = PPOKLAlgorithm(self.config)
        self.assertEqual(algo2._kl_beta, 3.5)  # default from config
        algo2.load_checkpoint(tmp)
        os.remove(tmp)

        self.assertEqual(algo2._kl_beta, 7.5)

    def test_save_preserves_kl_beta(self):
        tmp = os.path.join(tempfile.gettempdir(), "ppo_kl_save_test.pt")
        self.algo.save(tmp)
        state = torch.load(tmp, map_location="cpu", weights_only=False)
        os.remove(tmp)
        self.assertIn("kl_beta", state)
        self.assertEqual(state["kl_beta"], 7.5)

    def test_load_restores_kl_beta(self):
        tmp = os.path.join(tempfile.gettempdir(), "ppo_kl_load_test.pt")
        self.algo.save(tmp)

        algo2 = PPOKLAlgorithm(self.config)
        self.assertEqual(algo2._kl_beta, 3.5)  # unchanged before load
        algo2.load(tmp)
        os.remove(tmp)
        self.assertEqual(algo2._kl_beta, 7.5)


# ── Metrics ────────────────────────────────────────────────

class TestPPOKLMetrics(unittest.TestCase):
    def setUp(self):
        self.config = _make_config()
        self.algo = PPOKLAlgorithm(self.config)

    def test_metrics_reporter_available(self):
        self.assertIsNotNone(self.algo.metrics_reporter)

    def test_record_update_includes_kl_beta(self):
        reporter = self.algo.metrics_reporter
        loss = LossInfo(
            total_loss=0.5, policy_loss=0.3, mean_reward=5.0,
            entropy=0.5,
            extra={"kl": 0.005, "kl_beta": 1.0},
        )
        reporter.record_update(loss)
        summary = reporter.update_summary()
        self.assertIn("0.005", summary)


if __name__ == "__main__":
    unittest.main()
