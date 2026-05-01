"""PPO with Adaptive KL Penalty (PPO-KL).

Theory
------
PPO-Clip constrains the update by clipping the importance ratio:

    policy_loss = -min(ratio * A, clip(ratio, 1±ε) * A)

PPO-KL replaces the clipped policy objective with a soft penalty on
the KL divergence between the old and new policy:

    policy_loss = -(ratio * A)  +  β * KL[π_old || π_new]

The value loss still uses PPO-style clipping for stability.

The penalty coefficient β is adapted after each update:
    if KL < target_kl / 1.5  →  β ← β / 2
    if KL > target_kl * 1.5  →  β ← β * 2

This keeps the policy change near the target_kl threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from agent.base import ActResult, Algorithm, LossInfo
from agent.common.checkpoint import Checkpoint, capture_rng_state
from agent.nn import create_model
from agent.ppo.batch import compute_gae
from agent.ppo_kl.ppo_kl_metrics import PPOKLMetricsReporter
from agent.registry import register


def kl_div_categorical(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(π_old || π_new) per sample, for categorical distributions."""
    old_logp = F.log_softmax(old_logits, dim=-1)
    new_logp = F.log_softmax(new_logits, dim=-1)
    old_p = torch.exp(old_logp)
    return (old_p * (old_logp - new_logp)).sum(dim=-1)


@register("ppo_kl")
class PPOKLAlgorithm(Algorithm):
    """PPO with adaptive KL penalty (policy no clipping, value uses clipped objective)."""

    def __init__(self, config, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")

        self.model = create_model(
            getattr(config, "model_type", "shared"),
            config.num_actions,
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self._metrics_reporter = PPOKLMetricsReporter()

        self._kl_beta = float(config.kl_beta)

        # ── internal buffer ──────────────────────────────────
        self._map_imgs: list[np.ndarray] = []
        self._vectors: list[np.ndarray] = []
        self._legal_masks: list[np.ndarray] = []
        self._actions: list[int] = []
        self._old_log_probs: list[float] = []
        self._old_logits: list[np.ndarray] = []  # for KL computation
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

        self._last_logits: np.ndarray | None = None  # stashed by explore()

    # ── Metrics ─────────────────────────────────────────────

    @property
    def metrics_reporter(self) -> PPOKLMetricsReporter:
        return self._metrics_reporter

    # ── Tensor helpers ──────────────────────────────────────

    def _to_tensor(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # ── Action selection ────────────────────────────────────

    def explore(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, value, dist = self._run_model(map_img, vector, legal_mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # stash old logits for KL computation during update
        self._last_logits = logits.squeeze(0).detach().cpu().numpy()
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

    def compute_value(self, map_img: np.ndarray, vector: np.ndarray, legal_mask: np.ndarray) -> float:
        map_img_t, vector_t, legal_t = self._to_tensor(map_img, vector, legal_mask)
        with torch.no_grad():
            _, value = self.model(map_img_t, vector_t, legal_t)
        return value.item()

    # ── Experience collection ───────────────────────────────

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
        self._map_imgs.append(map_img)
        self._vectors.append(vector)
        self._legal_masks.append(np.array(legal_mask, dtype=np.float32))
        self._actions.append(action)
        self._old_log_probs.append(log_prob)
        self._old_logits.append(self._last_logits.copy())
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)

    def _buffer_clear(self) -> None:
        self._map_imgs.clear()
        self._vectors.clear()
        self._legal_masks.clear()
        self._actions.clear()
        self._old_log_probs.clear()
        self._old_logits.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()

    def ready_to_update(self) -> bool:
        return len(self._rewards) >= self.config.batch_size

    # ── Update ──────────────────────────────────────────────

    def update(self, bootstrap_value: float = 0.0) -> LossInfo:
        rewards = np.array(self._rewards, dtype=np.float32)
        values = np.array(self._values, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.int8)

        # 1. GAE
        advantages, returns = compute_gae(
            rewards, values, dones,
            self.config.gamma, self.config.gae_lambda, bootstrap_value,
        )
        mean_reward = float(rewards.mean())

        n = len(rewards)

        # 2. Stack batch tensors
        map_imgs_t = torch.as_tensor(
            np.stack(self._map_imgs), dtype=torch.float32, device=self.device,
        )
        vectors_t = torch.as_tensor(
            np.stack(self._vectors), dtype=torch.float32, device=self.device,
        )
        legal_masks_t = torch.as_tensor(
            np.stack(self._legal_masks), dtype=torch.float32, device=self.device,
        )
        actions_t = torch.as_tensor(
            np.array(self._actions, dtype=np.int64), device=self.device,
        )
        old_log_probs_t = torch.as_tensor(
            np.array(self._old_log_probs, dtype=np.float32), device=self.device,
        )
        old_logits_t = torch.as_tensor(
            np.stack(self._old_logits), dtype=torch.float32, device=self.device,
        )
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        self._buffer_clear()

        # 3. Multi-epoch mini-batch updates
        self.model.train()

        epoch_losses: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": [], "kl": [],
        }

        for _ in range(self.config.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.mini_batch_size):
                idx = indices[start:start + self.config.mini_batch_size]

                m_i = map_imgs_t[idx]
                v_i = vectors_t[idx]
                l_i = legal_masks_t[idx]
                a_i = actions_t[idx]
                old_lp_i = old_log_probs_t[idx]
                old_lo_i = old_logits_t[idx]
                adv_i = advantages_t[idx]
                ret_i = returns_t[idx]

                # Normalise advantages per mini-batch (PPO convention)
                adv_i = (adv_i - adv_i.mean()) / (adv_i.std() + 1e-8)

                logits, v_pred = self.model(m_i, v_i, l_i)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(a_i)
                entropy = dist.entropy()

                # Importance sampling ratio
                ratio = torch.exp(new_log_probs - old_lp_i)

                # KL penalty
                kl = kl_div_categorical(old_lo_i, logits)

                # Policy loss: surrogate + KL penalty (no clipping)
                surrogate = ratio * adv_i
                policy_loss = -(surrogate.mean()) + self._kl_beta * kl.mean()

                # Value loss (same as PPO, with clipping)
                old_v_i = torch.as_tensor(values[idx], dtype=torch.float32, device=self.device)
                value_pred_clipped = old_v_i + torch.clamp(
                    v_pred.squeeze(-1) - old_v_i,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon,
                )
                v_loss_unclipped = (v_pred.squeeze(-1) - ret_i) ** 2
                v_loss_clipped = (value_pred_clipped - ret_i) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                epoch_losses["policy_loss"].append(policy_loss.item())
                epoch_losses["value_loss"].append(value_loss.item())
                epoch_losses["entropy"].append(entropy.mean().item())
                epoch_losses["total_loss"].append(total_loss.item())
                epoch_losses["kl"].append(kl.mean().item())

        self.model.eval()

        # 4. Adaptive KL coefficient
        mean_kl = float(np.mean(epoch_losses["kl"]))
        if getattr(self.config, "kl_adaptive", True):
            if mean_kl < self.config.target_kl / 1.5:
                self._kl_beta /= 2.0
            elif mean_kl > self.config.target_kl * 1.5:
                self._kl_beta *= 2.0
            self._kl_beta = np.clip(self._kl_beta, 1e-10, 1e10)

        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        loss_info = LossInfo(
            total_loss=avg["total_loss"],
            policy_loss=avg["policy_loss"],
            value_loss=avg["value_loss"],
            entropy=avg["entropy"],
            mean_reward=mean_reward,
            extra={"kl": mean_kl, "kl_beta": self._kl_beta},
        )
        self._metrics_reporter.record_update(loss_info)
        return loss_info

    # ── Serialisation ───────────────────────────────────────

    def save(self, path: str | Path) -> None:
        state = {
            "model_state_dict": self.model.state_dict(),
            "kl_beta": self._kl_beta,
        }
        torch.save(state, str(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            self.model.load_state_dict(state["model_state_dict"])
            self._kl_beta = float(state.get("kl_beta", self.config.kl_beta))
        else:
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
        data = ckpt.to_dict()
        data["kl_beta"] = self._kl_beta
        torch.save(data, str(path))

    def load_checkpoint(self, path: str | Path) -> Checkpoint:
        data = torch.load(str(path), map_location=self.device, weights_only=False)
        ckpt = Checkpoint.from_dict(data)
        self.model.load_state_dict(ckpt.model_state_dict)
        self.model.to(self.device)
        self.optimizer.load_state_dict(ckpt.optimizer_state_dict)
        self._kl_beta = float(data.get("kl_beta", self.config.kl_beta))
        return ckpt
