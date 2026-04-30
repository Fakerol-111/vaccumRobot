from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import torch


CHECKPOINT_VERSION = "2.0"


@dataclass
class Checkpoint:
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    global_step: int
    episode_counter: int = 0
    current_map_idx: int = 0
    current_map_id: int = 0
    current_stage_name: str = ""

    config_snapshot: dict[str, Any] = field(default_factory=dict)

    rng_state: dict[str, Any] = field(default_factory=dict)

    format_version: str = CHECKPOINT_VERSION
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "timestamp": self.timestamp or datetime.now().isoformat(),
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "global_step": self.global_step,
            "episode_counter": self.episode_counter,
            "current_map_idx": self.current_map_idx,
            "current_map_id": self.current_map_id,
            "current_stage_name": self.current_stage_name,
            "config_snapshot": self.config_snapshot,
            "rng_state": self.rng_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        return cls(
            model_state_dict=data["model_state_dict"],
            optimizer_state_dict=data["optimizer_state_dict"],
            global_step=data["global_step"],
            episode_counter=data.get("episode_counter", 0),
            current_map_idx=data.get("current_map_idx", 0),
            current_map_id=data.get("current_map_id", 0),
            current_stage_name=data.get("current_stage_name", ""),
            config_snapshot=data.get("config_snapshot", {}),
            rng_state=data.get("rng_state", {}),
            format_version=data.get("format_version", "1.0"),
            timestamp=data.get("timestamp", ""),
        )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    else:
        state["torch_cuda"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "python" in state:
        random.setstate(state["python"])


def build_config_snapshot(config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "gae_lambda": getattr(config, "gae_lambda", None),
        "clip_epsilon": getattr(config, "clip_epsilon", None),
        "value_coef": getattr(config, "value_coef", None),
        "entropy_coef": getattr(config, "entropy_coef", None),
        "max_grad_norm": config.max_grad_norm,
        "ppo_epochs": getattr(config, "ppo_epochs", None),
        "batch_size": getattr(config, "batch_size", None),
        "mini_batch_size": getattr(config, "mini_batch_size", None),
        "total_timesteps": config.total_timesteps,
        "save_interval": config.save_interval,
        "log_interval": config.log_interval,
        "max_npcs": config.max_npcs,
        "local_view_size": config.local_view_size,
        "num_actions": config.num_actions,
    }
    if extra:
        base.update(extra)
    return base
