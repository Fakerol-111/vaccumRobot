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


def build_config_snapshot(trainer) -> dict[str, Any]:
    c = trainer.config
    return {
        "learning_rate": c.learning_rate,
        "gamma": c.gamma,
        "gae_lambda": c.gae_lambda,
        "clip_epsilon": c.clip_epsilon,
        "value_coef": c.value_coef,
        "entropy_coef": c.entropy_coef,
        "max_grad_norm": c.max_grad_norm,
        "ppo_epochs": c.ppo_epochs,
        "batch_size": c.batch_size,
        "mini_batch_size": c.mini_batch_size,
        "total_timesteps": c.total_timesteps,
        "save_interval": c.save_interval,
        "log_interval": c.log_interval,
        "max_npcs": c.max_npcs,
        "local_view_size": c.local_view_size,
        "num_actions": c.num_actions,
        "default_npc_count": trainer.default_npc_count,
        "default_station_count": trainer.default_station_count,
        "map_strategy": trainer.map_strategy,
        "curriculum_enabled": trainer.curriculum_enabled,
        "curriculum_stages": trainer.curriculum_stages,
        "default_map_list": trainer._default_map_list,
        "seed": trainer._base_seed,
    }
