from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agent.base import ActResult, Algorithm, LossInfo
from agent.common.checkpoint import Checkpoint, capture_rng_state
from agent.nn.actor_critic import ActorCritic
from agent.ppo.batch import RolloutBatch, compute_gae
from agent.ppo.buffer import RolloutBuffer
from agent.ppo.ppo_metrics import PPOMetricsReporter
from agent.ppo.update import PPO
from agent.registry import register


def _prepare_batches(segment: RolloutBatch, mini_batch_size: int) -> list[RolloutBatch]:
    """Shuffle and split a RolloutBatch into mini-batches (PPO internal)."""
    n = len(segment)
    indices = np.random.permutation(n)
    batches: list[RolloutBatch] = []
    for start in range(0, n, mini_batch_size):
        end = start + mini_batch_size
        idx = indices[start:end]
        batches.append(segment[idx])
    return batches


@register("ppo")
class PPOAlgorithm(Algorithm):
    """PPO-specific algorithm implementation."""

    def __init__(self, config, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")

        self.model = ActorCritic(num_actions=config.num_actions)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self._ppo = PPO(self.model, self.optimizer, config, self.device)
        self._buffer = RolloutBuffer()
        self._metrics_reporter = PPOMetricsReporter()

    # ── Algorithm interface ────────────────────────────────

    @property
    def metrics_reporter(self) -> PPOMetricsReporter:
        return self._metrics_reporter

    def _to_tensor(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize numpy inputs to batched tensors on the correct device.

        Input shapes (single/batch):
            map_img:   (C, H, W) or (B, C, H, W)
            vector:    (N,)     or (B, N)
            legal_mask:(A,)     or (B, A)
        Returns batched (4D / 2D) tensors.
        """
        map_img_t = torch.as_tensor(map_img, dtype=torch.float32, device=self.device)
        vector_t = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        legal_t = torch.as_tensor(legal_mask, dtype=torch.float32, device=self.device)
        if map_img_t.dim() == 3:
            map_img_t = map_img_t.unsqueeze(0)
            vector_t = vector_t.unsqueeze(0)
            legal_t = legal_t.unsqueeze(0)
        return map_img_t, vector_t, legal_t

    def _run_model(self, map_img, vector, legal_mask):
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            logits, value = self.model(map_img_t, vector_t, legal_t)
            dist = Categorical(logits=logits)
        return logits, value, dist

    def explore(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, value, dist = self._run_model(map_img, vector, legal_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return ActResult(action=action.item(), log_prob=log_prob.item(), value=value.item())

    def exploit(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, value, dist = self._run_model(map_img, vector, legal_mask)
        action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        return ActResult(action=action.item(), log_prob=log_prob.item(), value=value.item())

    def collect(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self._buffer.append(map_img, vector, legal_mask, action, log_prob, value, reward, done)

    def on_step(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> LossInfo | None:
        self._buffer.append(map_img, vector, legal_mask, action, log_prob, value, reward, done)
        return None

    def maybe_update(self, bootstrap_state: tuple | None = None) -> LossInfo | None:
        if len(self._buffer) < self.config.batch_size:
            return None

        rewards = np.array(self._buffer.rewards, dtype=np.float32)
        values = np.array(self._buffer.values, dtype=np.float32)
        dones = np.array(self._buffer.dones, dtype=np.int8)

        if bootstrap_state is not None:
            bs_img, bs_vec, bs_legal = bootstrap_state
            bs_legal = np.asarray(bs_legal, dtype=np.float32)
            bootstrap = self.compute_value(bs_img, bs_vec, bs_legal)
        else:
            bootstrap = 0.0

        advantages, returns = compute_gae(
            rewards, values, dones,
            self.config.gamma, self.config.gae_lambda, bootstrap,
        )

        rollout = self._buffer.to_batch(advantages, returns)
        mean_reward = float(np.mean(rewards))
        self._buffer.clear()

        self.model.train()
        epoch_losses: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
        }
        for _ in range(self.config.ppo_epochs):
            batches = _prepare_batches(rollout, self.config.mini_batch_size)
            for batch in batches:
                info = self._ppo.update(batch)
                for k, v in info.items():
                    epoch_losses[k].append(v)

        self.model.eval()
        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        loss_info = LossInfo(
            total_loss=avg["total_loss"],
            policy_loss=avg["policy_loss"],
            value_loss=avg["value_loss"],
            entropy=avg["entropy"],
            mean_reward=mean_reward,
        )
        self._metrics_reporter.record_update(loss_info)
        return loss_info

    def ready_to_update(self) -> bool:
        return len(self._buffer) >= self.config.batch_size

    def update(self, bootstrap_value: float = 0.0) -> LossInfo:
        rewards = np.array(self._buffer.rewards, dtype=np.float32)
        values = np.array(self._buffer.values, dtype=np.float32)
        dones = np.array(self._buffer.dones, dtype=np.int8)

        advantages, returns = compute_gae(
            rewards, values, dones,
            self.config.gamma, self.config.gae_lambda, bootstrap_value,
        )

        rollout = self._buffer.to_batch(advantages, returns)
        mean_reward = float(np.mean(rewards))
        self._buffer.clear()

        self.model.train()
        epoch_losses: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [],
        }
        for _ in range(self.config.ppo_epochs):
            batches = _prepare_batches(rollout, self.config.mini_batch_size)
            for batch in batches:
                info = self._ppo.update(batch)
                for k, v in info.items():
                    epoch_losses[k].append(v)

        self.model.eval()
        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        loss_info = LossInfo(
            total_loss=avg["total_loss"],
            policy_loss=avg["policy_loss"],
            value_loss=avg["value_loss"],
            entropy=avg["entropy"],
            mean_reward=mean_reward,
        )
        self._metrics_reporter.record_update(loss_info)
        return loss_info

    def compute_value(self, map_img: np.ndarray, vector: np.ndarray, legal_mask: np.ndarray) -> float:
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            _, value = self.model(map_img_t, vector_t, legal_t)
        return value.item()

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), str(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def save_checkpoint(
        self,
        path: str | Path,
        global_step: int,
        episode_counter: int = 0,
        current_map_idx: int = 0,
        current_map_id: int = 0,
        current_stage_name: str = "",
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        ckpt = Checkpoint(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            global_step=global_step,
            episode_counter=episode_counter,
            current_map_idx=current_map_idx,
            current_map_id=current_map_id,
            current_stage_name=current_stage_name,
            config_snapshot=config_snapshot or {},
            rng_state=capture_rng_state(),
        )
        torch.save(ckpt.to_dict(), str(path))

    def load_checkpoint(self, path: str | Path) -> Checkpoint:
        data = torch.load(str(path), map_location=self.device, weights_only=True)
        ckpt = Checkpoint.from_dict(data)
        self.model.load_state_dict(ckpt.model_state_dict)
        self.model.to(self.device)
        self.optimizer.load_state_dict(ckpt.optimizer_state_dict)
        return ckpt
