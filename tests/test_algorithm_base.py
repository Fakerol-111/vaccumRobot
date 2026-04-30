"""Algorithm 接口路由和 _to_tensor 测试。"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np
import torch

from agent.base import ActResult, Algorithm
from agent.ppo.algorithm import PPOAlgorithm


# ── helpers ────────────────────────────────────────────────

class _MockAlgorithm(Algorithm):
    """Minimal Algorithm subclass for testing act() routing."""

    def explore(self, map_img, vector, legal_mask) -> ActResult:
        return ActResult(action=0, log_prob=0.0, value=0.0)

    def exploit(self, map_img, vector, legal_mask) -> ActResult:
        return ActResult(action=1, log_prob=0.0, value=0.0)

    def collect(self, *args, **kwargs) -> None: ...
    def ready_to_update(self) -> bool: return False
    def update(self, bootstrap_value=0.0): ...
    def compute_value(self, *args) -> float: return 0.0
    def save(self, *args) -> None: ...
    def load(self, *args) -> None: ...
    def save_checkpoint(self, *args, **kwargs) -> None: ...
    def load_checkpoint(self, *args, **kwargs) -> Any: ...


class _MockConfig:
    num_actions = 6
    learning_rate = 0.0003
    clip_epsilon = 0.2
    value_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    ppo_epochs = 4
    mini_batch_size = 64
    batch_size = 2048
    gamma = 0.99
    gae_lambda = 0.95
    total_timesteps = 100000
    save_interval = 10000
    log_interval = 100
    max_npcs = 5
    local_view_size = 21


# ── act() routing tests ───────────────────────────────────

class TestActRouting(unittest.TestCase):
    def setUp(self):
        self.algo = _MockAlgorithm()
        self.map_img = np.zeros((3, 21, 21), dtype=np.float32)
        self.vector = np.zeros(10, dtype=np.float32)
        self.legal_mask = np.ones(6, dtype=np.float32)

    def test_act_explore_calls_explore(self):
        result = self.algo.act(self.map_img, self.vector, self.legal_mask, mode="explore")
        self.assertEqual(result.action, 0)

    def test_act_exploit_calls_exploit(self):
        result = self.algo.act(self.map_img, self.vector, self.legal_mask, mode="exploit")
        self.assertEqual(result.action, 1)

    def test_act_defaults_to_exploit(self):
        result = self.algo.act(self.map_img, self.vector, self.legal_mask)
        self.assertEqual(result.action, 1)

    def test_act_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.algo.act(self.map_img, self.vector, self.legal_mask, mode="invalid")  # type: ignore[arg-type]


# ── _to_tensor() tests ────────────────────────────────────

class TestToTensor(unittest.TestCase):
    def setUp(self):
        self.algo = PPOAlgorithm(_MockConfig(), device=torch.device("cpu"))

    def test_single_sample_unsqueezes(self):
        map_img = np.zeros((3, 21, 21), dtype=np.float32)     # (C, H, W)
        vector = np.zeros(10, dtype=np.float32)                # (N,)
        legal = np.ones(6, dtype=np.float32)                   # (A,)

        m, v, l = self.algo._to_tensor(map_img, vector, legal)

        self.assertEqual(m.shape, (1, 3, 21, 21))
        self.assertEqual(v.shape, (1, 10))
        self.assertEqual(l.shape, (1, 6))
        self.assertEqual(m.dtype, torch.float32)
        self.assertEqual(v.dtype, torch.float32)

    def test_batch_passthrough(self):
        map_img = np.zeros((4, 3, 21, 21), dtype=np.float32)  # (B, C, H, W)
        vector = np.zeros((4, 10), dtype=np.float32)           # (B, N)
        legal = np.ones((4, 6), dtype=np.float32)              # (B, A)

        m, v, l = self.algo._to_tensor(map_img, vector, legal)

        self.assertEqual(m.shape, (4, 3, 21, 21))
        self.assertEqual(v.shape, (4, 10))
        self.assertEqual(l.shape, (4, 6))

    def test_device_and_dtype(self):
        map_img = np.ones((3, 5, 5), dtype=np.float64)
        vector = np.ones(4, dtype=np.float64)
        legal = np.ones(3, dtype=np.float64)

        m, v, l = self.algo._to_tensor(map_img, vector, legal)

        self.assertEqual(m.device.type, "cpu")
        self.assertEqual(m.dtype, torch.float32)  # always cast to float32


if __name__ == "__main__":
    unittest.main()
