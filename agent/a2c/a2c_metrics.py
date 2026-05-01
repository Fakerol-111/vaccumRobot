"""A2C 算法监控指标。"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.base import LossInfo, MetricsReporter


class A2CMetricsReporter(MetricsReporter):
    """追踪 policy_loss / value_loss / entropy / mean_reward。"""

    def __init__(self, collector: Any = None) -> None:
        super().__init__(collector)
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropies: list[float] = []
        self.rewards: list[float] = []

    def record_update(self, loss_info: LossInfo) -> None:
        self.rewards.append(loss_info.mean_reward)
        self.policy_losses.append(loss_info.policy_loss or 0.0)
        self.value_losses.append(loss_info.value_loss or 0.0)
        self.entropies.append(loss_info.entropy or 0.0)
        self._push_event("update", {
            "reward": loss_info.mean_reward,
            "policy_loss": loss_info.policy_loss,
            "value_loss": loss_info.value_loss,
            "entropy": loss_info.entropy,
        })

    def update_summary(self) -> str:
        n = 100
        pl = np.mean(self.policy_losses[-n:]) if self.policy_losses else 0.0
        vl = np.mean(self.value_losses[-n:]) if self.value_losses else 0.0
        en = np.mean(self.entropies[-n:]) if self.entropies else 0.0
        return f"policy_loss={pl:.4f}  value_loss={vl:.4f}  entropy={en:.4f}"

    def final_summary_lines(self) -> list[str]:
        if not self.rewards:
            return ["(no updates recorded)"]
        return [
            f"Updates: {len(self.rewards)}",
            f"policy_loss={np.mean(self.policy_losses):.4f}  value_loss={np.mean(self.value_losses):.4f}",
            f"entropy={np.mean(self.entropies):.4f}  mean_reward={np.mean(self.rewards):.4f}",
        ]
