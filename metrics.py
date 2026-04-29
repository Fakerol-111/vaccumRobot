from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from training_dashboard import MetricsCollector


class MetricsLogger:
    def __init__(self, log_file: Path | None = None, collector: MetricsCollector | None = None):
        self.cleaned_list: list[int] = []
        self.steps_list: list[int] = []
        self.charge_list: list[int] = []
        self.reward_list: list[float] = []

        self.update_rewards: list[float] = []
        self.policy_losses: list[float] = []
        self.value_losses: list[float] = []
        self.entropies: list[float] = []

        self._ema_alpha = 2.0 / 101.0
        self._ema_cleaned = 0.0

        self._collector = collector

        self._log_file = log_file
        if self._log_file is not None:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = open(self._log_file, "w", encoding="utf-8")
        else:
            self._file_handle = None

    # ---------- episode ----------

    def log_episode(
        self,
        cleaned: int,
        steps: int,
        charge_count: int,
        reward_sum: float,
        map_name: str = "",
    ) -> None:
        self.cleaned_list.append(cleaned)
        self.steps_list.append(steps)
        self.charge_list.append(charge_count)
        self.reward_list.append(reward_sum)

        if self.episode_count == 1:
            self._ema_cleaned = float(cleaned)
        else:
            self._ema_cleaned = self._ema_alpha * float(cleaned) + (1 - self._ema_alpha) * self._ema_cleaned

        efficiency = cleaned / max(steps, 1)

        map_tag = f"[{map_name}] " if map_name else ""
        msg = (
            f"[Episode {self.episode_count:>4d}] "
            f"{map_tag}"
            f"cleaned={cleaned:>5d}  steps={steps:>4d}  "
            f"charges={charge_count:>2d}  "
            f"reward={reward_sum:>7.2f}  "
            f"ema_cleaned={self._ema_cleaned:>6.1f}  "
            f"eff={efficiency:.3f}"
        )
        self._emit(msg)

        if self._collector is not None:
            self._collector.add_event("episode", {
                "episode": self.episode_count,
                "map_name": map_name,
                "cleaned": cleaned,
                "steps": steps,
                "charges": charge_count,
                "reward": reward_sum,
                "ema_cleaned": self._ema_cleaned,
                "efficiency": efficiency,
            })

    # ---------- update ----------

    def log_update(self, reward: float, policy_loss: float, value_loss: float, entropy: float) -> None:
        self.update_rewards.append(reward)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)

        if self._collector is not None:
            self._collector.add_event("update", {
                "update_idx": self.update_count,
                "reward": reward,
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
            })

    # ---------- periodic log ----------

    def print_summary(self, step: int, fps: float | None = None) -> None:
        u = self._update_stats()

        parts = [f"step={step}"]
        if fps is not None:
            parts.append(f"fps={fps:.0f}")
        parts.append(
            f"policy_loss={u['policy_loss']['mean']:.4f} "
            f"value_loss={u['value_loss']['mean']:.4f} "
            f"entropy={u['entropy']['mean']:.4f}"
        )
        parts.append(
            f"ema_cleaned={self._ema_cleaned:.1f}  "
            f"episodes={self.episode_count}"
        )
        self._emit("  ".join(parts))

        if self._collector is not None:
            self._collector.add_event("summary", {
                "step": step,
                "fps": fps,
                "policy_loss": u["policy_loss"]["mean"],
                "value_loss": u["value_loss"]["mean"],
                "entropy": u["entropy"]["mean"],
                "ema_cleaned": self._ema_cleaned,
                "episodes": self.episode_count,
            })

    # ---------- end of training ----------

    def print_training_summary(self) -> None:
        self._emit("")
        self._emit("=== Training Summary ===")

        c = self._episode_stats(self.cleaned_list)
        s = self._episode_stats(self.steps_list)
        ch = self._episode_stats(self.charge_list)
        r = self._episode_stats_float(self.reward_list)

        self._emit(
            f"Episodes: {self.episode_count}"
        )
        self._emit(
            f"Cleaned:  avg={c['mean']:.1f} std={c['std']:.1f} "
            f"min={c['min']:.0f} max={c['max']:.0f}"
        )
        self._emit(
            f"Steps:    avg={s['mean']:.1f} std={s['std']:.1f} "
            f"min={s['min']:.0f} max={s['max']:.0f}"
        )
        self._emit(
            f"Charges:  avg={ch['mean']:.2f} std={ch['std']:.2f} "
            f"min={ch['min']:.0f} max={ch['max']:.0f}"
        )
        self._emit(
            f"Efficiency: avg={(c['sum'] / max(s['sum'], 1)):.3f}"
        )
        self._emit(
            f"Reward:   avg={r['mean']:.2f} std={r['std']:.2f}"
        )

        u = self._update_stats()
        self._emit(
            f"Updates:  {self.update_count}  "
            f"policy_loss={u['policy_loss']['mean']:.4f}  "
            f"value_loss={u['value_loss']['mean']:.4f}  "
            f"entropy={u['entropy']['mean']:.4f}  "
            f"mean_reward={u['reward']['mean']:.4f}"
        )

    # ---------- properties ----------

    @property
    def episode_count(self) -> int:
        return len(self.cleaned_list)

    @property
    def update_count(self) -> int:
        return len(self.update_rewards)

    # ---------- internal ----------

    @staticmethod
    def _episode_stats(values: list[int]) -> dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sum": float(np.sum(arr)),
        }

    @staticmethod
    def _episode_stats_float(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "sum": float(np.sum(arr)),
        }

    def _update_stats(self) -> dict[str, dict[str, float]]:
        def _s(values: list[float]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0}
            arr = np.array(values, dtype=np.float32)
            return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

        return {
            "reward": _s(self.update_rewards),
            "policy_loss": _s(self.policy_losses),
            "value_loss": _s(self.value_losses),
            "entropy": _s(self.entropies),
        }

    def _emit(self, msg: str) -> None:
        print(msg, flush=True)
        if self._file_handle is not None:
            self._file_handle.write(msg + "\n")
            self._file_handle.flush()

    def close(self) -> None:
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
