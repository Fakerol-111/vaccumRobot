"""GRPO 算法监控指标。"""

from __future__ import annotations

from typing import Any

import numpy as np

from agent.base import LossInfo, MetricsReporter


class GRPOMetricsReporter(MetricsReporter):
    """追踪 total_loss / policy_loss / mean_score / std_score / kl / entropy。"""

    def __init__(self, collector: Any = None) -> None:
        super().__init__(collector)
        self.total_losses: list[float] = []
        self.policy_losses: list[float] = []
        self.mean_scores: list[float] = []
        self.std_scores: list[float] = []
        self.kls: list[float] = []
        self.entropies: list[float] = []

    def record_update(self, loss_info: LossInfo) -> None:
        extra = loss_info.extra
        self.total_losses.append(loss_info.total_loss)
        self.policy_losses.append(loss_info.policy_loss or 0.0)
        self.mean_scores.append(loss_info.mean_reward)
        self.std_scores.append(extra.get("std_reward", 0.0))
        self.kls.append(extra.get("kl_divergence", 0.0))
        self.entropies.append(loss_info.entropy or 0.0)
        self._push_event("group_update", {
            "total_loss": loss_info.total_loss,
            "policy_loss": loss_info.policy_loss,
            "mean_score": loss_info.mean_reward,
            "std_score": extra.get("std_reward"),
            "kl": extra.get("kl_divergence"),
            "entropy": loss_info.entropy,
        })

    def update_summary(self) -> str:
        n = 50
        ms = np.mean(self.mean_scores[-n:]) if self.mean_scores else 0.0
        ss = np.mean(self.std_scores[-n:]) if self.std_scores else 0.0
        kl = np.mean(self.kls[-n:]) if self.kls else 0.0
        en = np.mean(self.entropies[-n:]) if self.entropies else 0.0
        return f"mean_score={ms:.2f}  std_score={ss:.2f}  kl={kl:.4f}  entropy={en:.4f}"

    def final_summary_lines(self) -> list[str]:
        if not self.total_losses:
            return ["(no group updates recorded)"]
        return [
            f"Group updates: {len(self.total_losses)}",
            f"mean_score={np.mean(self.mean_scores):.2f} ± {np.mean(self.std_scores):.2f}",
            f"kl={np.mean(self.kls):.4f}  entropy={np.mean(self.entropies):.4f}",
            f"total_loss={np.mean(self.total_losses):.4f}",
        ]
