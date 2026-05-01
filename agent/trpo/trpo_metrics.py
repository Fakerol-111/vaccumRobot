"""TRPO 监控指标。"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.base import LossInfo, MetricsReporter


class TRPOMetricsReporter(MetricsReporter):
    """追踪 surrogate_loss / value_loss / entropy / kl / max_kl / mean_reward。"""

    def __init__(self, collector: Any = None) -> None:
        super().__init__(collector)
        self.surrogate_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropies: list[float] = []
        self.kl_divergences: list[float] = []
        self.max_kls: list[float] = []
        self.rewards: list[float] = []

    def record_update(self, loss_info: LossInfo) -> None:
        self.rewards.append(loss_info.mean_reward)
        self.surrogate_losses.append(loss_info.extra.get("surrogate_loss", 0.0))
        self.value_losses.append(loss_info.value_loss or 0.0)
        self.entropies.append(loss_info.entropy or 0.0)
        self.kl_divergences.append(loss_info.extra.get("kl", 0.0))
        self.max_kls.append(loss_info.extra.get("max_kl", 0.0))
        self._push_event("update", {
            "reward": loss_info.mean_reward,
            "surrogate_loss": loss_info.extra.get("surrogate_loss"),
            "value_loss": loss_info.value_loss,
            "entropy": loss_info.entropy,
            "kl": loss_info.extra.get("kl"),
            "max_kl": loss_info.extra.get("max_kl"),
        })

    def update_summary(self) -> str:
        n = 100
        sl = np.mean(self.surrogate_losses[-n:]) if self.surrogate_losses else 0.0
        vl = np.mean(self.value_losses[-n:]) if self.value_losses else 0.0
        en = np.mean(self.entropies[-n:]) if self.entropies else 0.0
        kl = np.mean(self.kl_divergences[-n:]) if self.kl_divergences else 0.0
        return f"surrogate_loss={sl:.4f}  value_loss={vl:.4f}  entropy={en:.4f}  kl={kl:.6f}"

    def final_summary_lines(self) -> list[str]:
        if not self.rewards:
            return ["(no updates recorded)"]
        return [
            f"Updates: {len(self.rewards)}",
            f"surrogate_loss={np.mean(self.surrogate_losses):.4f}  value_loss={np.mean(self.value_losses):.4f}",
            f"entropy={np.mean(self.entropies):.4f}  kl={np.mean(self.kl_divergences):.6f}",
            f"mean_reward={np.mean(self.rewards):.4f}",
        ]
