"""A2C (Advantage Actor-Critic) with n-step returns.

Theory
------
A2C sits between REINFORCE (pure MC) and PPO (GAE+clip) on the variance-bias spectrum.

For each step t, the n-step return bootstraps after looking *n* steps ahead:

    G_t^(n) = sum_{k=0}^{n-1} gamma^k * r_{t+k}  +  gamma^n * V(s_{t+n})

    advantage_t = G_t^(n) - V(s_t)

- n=1:      TD(0), highest bias, lowest variance
- n=large:  approaches MC return (REINFORCE-like), lower bias, higher variance

Losses:
    policy_loss = -log_prob(a_t|s_t) * advantage_t.detach()
    value_loss  = MSE(V(s_t), G_t^(n))
    total       = policy_loss + value_coef * value_loss - entropy_coef * H(pi)
"""

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
from agent.nn import create_model
from agent.registry import register
from agent.a2c.a2c_metrics import A2CMetricsReporter


def compute_n_step_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    n_step: int,
    bootstrap_value: float = 0.0,
) -> np.ndarray:
    """N-step return with bootstrapping.

    For each step t:
        G_t = gamma^0 * r_t + gamma^1 * r_{t+1} + ...
              + gamma^(n-1) * r_{t+n-1}  +  gamma^n * V(s_{t+n})

    - Episode termination (``done=True``) cuts the accumulation short
      (pure MC for that segment, no bootstrap across episode boundary).
    - Past the buffer boundary, *bootstrap_value* is used as continuation.
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)

    for t in range(T):
        G = 0.0
        gk = 1.0

        for k in range(n_step):
            idx = t + k
            if idx >= T:
                G += gk * bootstrap_value
                break
            G += gk * rewards[idx]
            gk *= gamma
            if dones[idx]:
                break
        else:
            # completed n steps without hitting done or T
            nxt = t + n_step
            if nxt < T:
                G += gk * values[nxt]
            elif nxt == T:
                G += gk * bootstrap_value

        returns[t] = G

    return returns


@register("a2c")
class A2CAlgorithm(Algorithm):
    """Advantage Actor-Critic with n-step returns.

    Collects on-policy experience in a buffer. When the buffer reaches
    ``batch_size`` steps, computes n-step returns and performs a single
    gradient update (no multi-epoch, no clipping).
    """

    def __init__(self, config, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")

        self.model = create_model(
            getattr(config, "model_type", "shared"),
            config.num_actions,
        )
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self._metrics_reporter = A2CMetricsReporter()

        # ── internal buffer ──────────────────────────────────
        self._map_imgs: list[np.ndarray] = []
        self._vectors: list[np.ndarray] = []
        self._legal_masks: list[np.ndarray] = []
        self._actions: list[int] = []
        self._log_probs: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[bool] = []

    # ── Metrics ─────────────────────────────────────────────

    @property
    def metrics_reporter(self) -> A2CMetricsReporter:
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
        return ActResult(
            action=action.item(), log_prob=log_prob.item(), value=value.item()
        )

    def exploit(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        logits, value, dist = self._run_model(map_img, vector, legal_mask)
        action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        return ActResult(
            action=action.item(), log_prob=log_prob.item(), value=value.item()
        )

    def compute_value(
        self, map_img: np.ndarray, vector: np.ndarray, legal_mask: np.ndarray
    ) -> float:
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
        self._log_probs.append(log_prob)
        self._values.append(value)
        self._rewards.append(reward)
        self._dones.append(done)

    def _buffer_clear(self) -> None:
        self._map_imgs.clear()
        self._vectors.clear()
        self._legal_masks.clear()
        self._actions.clear()
        self._log_probs.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()

    def ready_to_update(self) -> bool:
        return len(self._rewards) >= self.config.batch_size

    # ── Update ──────────────────────────────────────────────

    def update(self, bootstrap_value: float = 0.0) -> LossInfo:
        rewards = np.array(self._rewards, dtype=np.float32)
        dones = np.array(self._dones, dtype=np.int8)
        values = np.array(self._values, dtype=np.float32)

        # 1. N-step returns
        returns = compute_n_step_returns(
            rewards,
            values,
            dones,
            self.config.gamma,
            self.config.n_step,
            bootstrap_value,
        )
        advantages = returns - values
        mean_reward = float(rewards.mean())

        # 2. Stack batch tensors
        map_imgs_t = torch.as_tensor(
            np.stack(self._map_imgs),
            dtype=torch.float32,
            device=self.device,
        )
        vectors_t = torch.as_tensor(
            np.stack(self._vectors),
            dtype=torch.float32,
            device=self.device,
        )
        legal_masks_t = torch.as_tensor(
            np.stack(self._legal_masks),
            dtype=torch.float32,
            device=self.device,
        )
        actions_t = torch.as_tensor(
            np.array(self._actions, dtype=np.int64),
            device=self.device,
        )
        advantages_t = torch.as_tensor(
            advantages, dtype=torch.float32, device=self.device
        )
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        self._buffer_clear()

        # 3. Advantage normalisation
        if getattr(self.config, "normalize_advantage", True) and len(advantages_t) > 1:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        # 4. Forward pass
        self.model.train()
        logits, values_pred = self.model(map_imgs_t, vectors_t, legal_masks_t)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        # 5. Losses (same structure as REINFORCE, different source of returns)
        policy_loss = -(log_probs * advantages_t.detach()).mean()
        value_loss = nn.MSELoss()(values_pred.squeeze(-1), returns_t.detach())

        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self.model.eval()

        loss_info = LossInfo(
            total_loss=total_loss.item(),
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            mean_reward=mean_reward,
        )
        self._metrics_reporter.record_update(loss_info)
        return loss_info

    # ── Serialisation ───────────────────────────────────────

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
        data = torch.load(str(path), map_location=self.device, weights_only=False)
        ckpt = Checkpoint.from_dict(data)
        self.model.load_state_dict(ckpt.model_state_dict)
        self.model.to(self.device)
        self.optimizer.load_state_dict(ckpt.optimizer_state_dict)
        return ckpt
