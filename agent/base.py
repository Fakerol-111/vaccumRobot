from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class ActResult:
    """Result of a single action selection."""
    action: int
    log_prob: float | None = None
    value: float | None = None


@dataclass
class LossInfo:
    """Loss statistics returned by algorithm.update().

    Common fields (present for all algorithms):

        total_loss   — combined loss value
        policy_loss  — policy gradient component
        value_loss   — value function component (PPO, etc.)
        entropy      — policy entropy (for monitoring)
        mean_reward  — average reward in the batch

    Algorithm-specific fields live in *extra* to keep the base lean.
    """
    total_loss: float
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    mean_reward: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


class Algorithm(ABC):
    """Unified reinforcement learning algorithm interface.

    The algorithm owns its internal experience buffer and update logic.
    The trainer/environment loop calls ``act`` → ``collect`` repeatedly,
    then ``update`` when the buffer is full.

    Two action modes are supported:
        "explore" — stochastic action sampling (for training).
        "exploit" — deterministic action selection (for evaluation).
    """

    def act(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
        mode: Literal["explore", "exploit"] = "exploit",
    ) -> ActResult:
        if mode == "explore":
            return self.explore(map_img, vector, legal_mask)
        elif mode == "exploit":
            return self.exploit(map_img, vector, legal_mask)
        else:
            raise ValueError(f"Unknown action mode: {mode!r} (expected 'explore' or 'exploit')")

    @abstractmethod
    def explore(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        """Choose action stochastically for exploration (training)."""
        ...

    @abstractmethod
    def exploit(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
    ) -> ActResult:
        """Choose action deterministically for exploitation (evaluation)."""
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def ready_to_update(self) -> bool:
        ...

    @abstractmethod
    def update(self, bootstrap_value: float = 0.0) -> LossInfo:
        ...

    @abstractmethod
    def compute_value(self, map_img: np.ndarray, vector: np.ndarray, legal_mask: np.ndarray) -> float:
        ...

    @abstractmethod
    def save(self, path: str | Path) -> None:
        ...

    @abstractmethod
    def load(self, path: str | Path) -> None:
        ...

    @abstractmethod
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
        ...

    @abstractmethod
    def load_checkpoint(self, path: str | Path) -> Any:
        ...

    # ── unified lifecycle hooks ───────────────────────────

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
        """Called every step. Return LossInfo if the algorithm performed an update."""
        self.collect(map_img, vector, legal_mask, action, log_prob, value, reward, done)
        return None

    def maybe_update(self, bootstrap_state: tuple | None = None) -> LossInfo | None:
        """Called after a batch of steps. Return LossInfo if an update ran.

        Args:
            bootstrap_state: Optional (map_img, vector, legal_mask) for computing
                            the bootstrapped value of the final state.
        """
        if not self.ready_to_update():
            return None
        if bootstrap_state is not None:
            bs_img, bs_vec, bs_legal = bootstrap_state
            bootstrap = self.compute_value(bs_img, bs_vec, np.asarray(bs_legal, dtype=np.float32))
        else:
            bootstrap = 0.0
        return self.update(bootstrap)

    # ── optional hooks ────────────────────────────────────

    def set_env_config(self, env_config: dict[str, Any]) -> None:
        """可选：通知算法当前环境的配置（地图参数、NPC 数量等）。

        在环境切换时由 Trainer 调用。默认无操作。
        """
        return

    @property
    def metrics_reporter(self) -> MetricsReporter | None:
        """各算法自定义的监控指标采集器。"""
        return None



class MetricsReporter(ABC):
    """算法自定义监控指标采集器。

    每个算法在对应文件夹下实现子类，定义需要追踪的指标、
    格式化方式和 dashboard 事件推送。
    """

    def __init__(self, collector: Any = None) -> None:
        self._collector = collector

    def set_collector(self, collector: Any) -> None:
        self._collector = collector

    @abstractmethod
    def record_update(self, loss_info: LossInfo) -> None:
        """记录一次更新步骤的指标。"""

    @abstractmethod
    def update_summary(self) -> str:
        """周期日志的一行摘要（不含 step/fps 前缀）。"""

    @abstractmethod
    def final_summary_lines(self) -> list[str]:
        """训练结束时的汇总行。"""

    def _push_event(self, event_type: str, data: dict) -> None:
        if self._collector is not None:
            self._collector.add_event(event_type, data)
