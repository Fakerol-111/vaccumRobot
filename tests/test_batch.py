"""GAE 计算测试：agent/ppo/batch.py"""

from __future__ import annotations

import unittest

import numpy as np

from agent.ppo.batch import RolloutBatch, compute_gae


class TestComputeGAE(unittest.TestCase):
    """compute_gae with known inputs/outputs."""

    def test_single_step_no_discount(self):
        rewards = np.array([1.0], dtype=np.float32)
        values = np.array([0.5], dtype=np.float32)
        dones = np.array([0], dtype=np.int8)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.0, gae_lambda=0.0)
        np.testing.assert_allclose(adv, [0.5], atol=1e-6)
        np.testing.assert_allclose(ret, [1.0], atol=1e-6)

    def test_multi_step_discount(self):
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        values = np.array([0.9, 0.9, 0.9], dtype=np.float32)
        dones = np.array([0, 0, 0], dtype=np.int8)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.9, gae_lambda=0.95)
        # t=2: delta = 1 + 0.9*0 - 0.9 = 0.1, adv[2] = 0.1
        # t=1: delta = 1 + 0.9*0.9 - 0.9 = 0.91, adv[1] = 0.91 + 0.9*0.95*0.1 = 0.9955
        # t=0: delta = 1 + 0.9*0.9 - 0.9 = 0.91, adv[0] = 0.91 + 0.9*0.95*0.9955 = 1.76115
        np.testing.assert_allclose(adv[0], 1.76115, atol=1e-4)
        np.testing.assert_allclose(adv[1], 0.9955, atol=1e-4)
        np.testing.assert_allclose(adv[2], 0.1, atol=1e-6)
        np.testing.assert_allclose(ret[0], 2.66115, atol=1e-4)
        np.testing.assert_allclose(ret[1], 1.8955, atol=1e-4)
        np.testing.assert_allclose(ret[2], 1.0, atol=1e-6)

    def test_terminal_done_stops_bootstrap(self):
        rewards = np.array([1.0, 5.0], dtype=np.float32)
        values = np.array([2.0, 3.0], dtype=np.float32)
        dones = np.array([0, 1], dtype=np.int8)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.9, gae_lambda=0.95)
        # t=1 (done): delta = 5 + 0.9*0*(1-1) - 3 = 2, adv[1] = 2
        # t=0: delta = 1 + 0.9*3*(1-0) - 2 = 1.7
        #       adv[0] = 1.7 + 0.9*0.95*(1-0)*2 = 1.7 + 1.71 = 3.41
        np.testing.assert_allclose(adv[1], 2.0, atol=1e-6)
        np.testing.assert_allclose(adv[0], 3.41, atol=1e-5)
        np.testing.assert_allclose(ret[1], 5.0, atol=1e-6)
        np.testing.assert_allclose(ret[0], 5.41, atol=1e-5)

    def test_bootstrap_value(self):
        rewards = np.array([1.0], dtype=np.float32)
        values = np.array([0.0], dtype=np.float32)
        dones = np.array([0], dtype=np.int8)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.9, gae_lambda=1.0, bootstrap_value=10.0)
        # delta = 1 + 0.9*10*1 - 0 = 10
        # adv = 10
        # ret = 10 + 0 = 10
        np.testing.assert_allclose(adv, [10.0], atol=1e-5)
        np.testing.assert_allclose(ret, [10.0], atol=1e-5)

    def test_zero_rewards(self):
        rewards = np.zeros(5, dtype=np.float32)
        values = np.ones(5, dtype=np.float32)
        dones = np.zeros(5, dtype=np.int8)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        self.assertEqual(adv.shape, (5,))
        self.assertEqual(ret.shape, (5,))

    def test_known_values(self):
        rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        dones = np.array([0, 0, 0], dtype=np.int8)
        adv, ret = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
        # GAE formula reverse iteration:
        # t=2: delta = 1 + 0.99*0*(1-0) - 0.7 = 0.3
        #       adv[2] = 0.3 + 0.99*0.95*(1-0)*0 = 0.3
        # t=1: delta = 0 + 0.99*0.7*(1-0) - 0.6 = 0.093
        #       adv[1] = 0.093 + 0.99*0.95*(1-0)*0.3 = 0.093 + 0.28215 = 0.37515
        # t=0: delta = 0 + 0.99*0.6*(1-0) - 0.5 = 0.094
        #       adv[0] = 0.094 + 0.99*0.95*(1-0)*0.37515 = 0.094 + 0.35283 = 0.44683
        expected_adv = np.array([0.44683, 0.37515, 0.3], dtype=np.float32)
        expected_ret = np.array([0.94683, 0.97515, 1.0], dtype=np.float32)
        np.testing.assert_allclose(adv, expected_adv, atol=1e-4)
        np.testing.assert_allclose(ret, expected_ret, atol=1e-4)


class TestRolloutBatch(unittest.TestCase):
    def test_len(self):
        batch = RolloutBatch(
            map_imgs=np.zeros((3, 4, 21, 21), dtype=np.float32),
            vectors=np.zeros((3, 10), dtype=np.float32),
            legal_masks=np.zeros((3, 6), dtype=np.float32),
            actions=np.zeros(3, dtype=np.int64),
            log_probs=np.zeros(3, dtype=np.float32),
            values=np.zeros(3, dtype=np.float32),
            rewards=np.zeros(3, dtype=np.float32),
            dones=np.zeros(3, dtype=np.int8),
            advantages=np.zeros(3, dtype=np.float32),
            returns=np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(len(batch), 3)

    def test_getitem_single(self):
        batch = RolloutBatch(
            map_imgs=np.arange(6, dtype=np.float32).reshape(2, 1, 1, 3),
            vectors=np.arange(4, dtype=np.float32).reshape(2, 2),
            legal_masks=np.arange(4, dtype=np.float32).reshape(2, 2),
            actions=np.array([7, 8], dtype=np.int64),
            log_probs=np.array([0.1, 0.2], dtype=np.float32),
            values=np.array([0.5, 0.6], dtype=np.float32),
            rewards=np.array([1.0, 2.0], dtype=np.float32),
            dones=np.array([0, 1], dtype=np.int8),
            advantages=np.array([0.3, 0.4], dtype=np.float32),
            returns=np.array([1.3, 2.4], dtype=np.float32),
        )
        sub = batch[0]
        self.assertIsInstance(sub, RolloutBatch)
        self.assertEqual(len(sub), 1)

    def test_concatenate(self):
        def _make(n: int, base: float) -> RolloutBatch:
            return RolloutBatch(
                map_imgs=np.zeros((n, 1, 3, 3), dtype=np.float32),
                vectors=np.full((n, 2), base, dtype=np.float32),
                legal_masks=np.zeros((n, 4), dtype=np.float32),
                actions=np.zeros(n, dtype=np.int64),
                log_probs=np.zeros(n, dtype=np.float32),
                values=np.full(n, base, dtype=np.float32),
                rewards=np.full(n, base, dtype=np.float32),
                dones=np.zeros(n, dtype=np.int8),
                advantages=np.full(n, base, dtype=np.float32),
                returns=np.full(n, base, dtype=np.float32),
            )

        b1 = _make(2, 1.0)
        b2 = _make(3, 2.0)
        merged = RolloutBatch.concatenate([b1, b2])
        self.assertEqual(len(merged), 5)
        np.testing.assert_array_equal(merged.values[:2], [1.0, 1.0])
        np.testing.assert_array_equal(merged.values[2:], [2.0, 2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
