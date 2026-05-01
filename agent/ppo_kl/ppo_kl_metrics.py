"""PPO with Adaptive KL Penalty (PPO-KL) 监控指标。"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.base import LossInfo, MetricsReporter


class PPOKLMetricsReporter(MetricsReporter):
    """追踪 policy_loss / value_loss / entropy / kl_divergence / kl_beta / mean_reward。"""

    def __init__(self, collector: Any = None) -> None:
        super().__init__(collector)
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropies: list[float] = []
        self.kl_divergences: list[float] = []
        self.kl_betas: list[float] = []
        self.rewards: list[float] = []

    def record_update(self, loss_info: LossInfo) -> None:
        self.rewards.append(loss_info.mean_reward)
        self.policy_losses.append(loss_info.policy_loss or 0.0)
        self.value_losses.append(loss_info.value_loss or 0.0)
        self.entropies.append(loss_info.entropy or 0.0)
        self.kl_divergences.append(loss_info.extra.get("kl", 0.0))
        self.kl_betas.append(loss_info.extra.get("kl_beta", 0.0))
        self._push_event("update", {
            "reward": loss_info.mean_reward,
            "policy_loss": loss_info.policy_loss,
            "value_loss": loss_info.value_loss,
            "entropy": loss_info.entropy,
            "kl": loss_info.extra.get("kl"),
            "kl_beta": loss_info.extra.get("kl_beta"),
        })

    def update_summary(self) -> str:
        n = 100
        pl = np.mean(self.policy_losses[-n:]) if self.policy_losses else 0.0
        vl = np.mean(self.value_losses[-n:]) if self.value_losses else 0.0
        en = np.mean(self.entropies[-n:]) if self.entropies else 0.0
        kl = np.mean(self.kl_divergences[-n:]) if self.kl_divergences else 0.0
        kb = self.kl_betas[-1] if self.kl_betas else 0.0
        return f"policy_loss={pl:.4f}  value_loss={vl:.4f}  entropy={en:.4f}  kl={kl:.6f}  β={kb:.4f}"

    def final_summary_lines(self) -> list[str]:
        if not self.rewards:
            return ["(no updates recorded)"]
        return [
            f"Updates: {len(self.rewards)}",
            f"policy_loss={np.mean(self.policy_losses):.4f}  value_loss={np.mean(self.value_losses):.4f}",
            f"entropy={np.mean(self.entropies):.4f}  kl={np.mean(self.kl_divergences):.6f}",
            f"mean_reward={np.mean(self.rewards):.4f}",
        ]
