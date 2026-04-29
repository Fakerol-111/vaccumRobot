"""入口模块：读取配置文件，组装 Trainer 并启动训练。

通过 `python train_sync.py` 或 `python main.py` 直接运行。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from trainer import Trainer

PROJECT_ROOT = Path(__file__).resolve().parent


def _default_config_path() -> Path:
    return PROJECT_ROOT / "config" / "train_config.toml"


def load_ppo_config(config_path: Path | None = None) -> SimpleNamespace:
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    ppo = raw["ppo"]
    num_actors = int(ppo.get("num_actors", 4))
    max_npcs = int(ppo.get("max_npcs", 5))
    local_view_size = int(ppo.get("local_view_size", 21))
    num_actions = int(ppo.get("num_actions", 8))
    return SimpleNamespace(
        num_actors=num_actors,
        num_env_steps=int(ppo.get("num_env_steps", 128)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_epsilon=float(ppo.get("clip_epsilon", 0.2)),
        value_coef=float(ppo.get("value_coef", 0.5)),
        entropy_coef=float(ppo.get("entropy_coef", 0.01)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        ppo_epochs=int(ppo.get("ppo_epochs", 10)),
        batch_size=int(ppo.get("batch_size", 512)),
        mini_batch_size=int(ppo.get("mini_batch_size", 128)),
        total_timesteps=int(ppo.get("total_timesteps", 10_000)),
        save_interval=int(ppo.get("save_interval", 5_000)),
        log_interval=int(ppo.get("log_interval", 500)),
        max_npcs=max_npcs,
        local_view_size=local_view_size,
        num_actions=num_actions,
    )


def load_env_config(config_path: Path | None = None) -> tuple[dict[str, Any], str]:
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    env_section = raw["env"]
    map_name = env_section["map"]

    if map_name == "simple":
        from config.simple_map_config import SIMPLE_MAP_CONFIG
        kwargs = dict(SIMPLE_MAP_CONFIG)
    else:
        raise ValueError(f"Unknown map: {map_name}")

    for key in ("npc_count", "station_count"):
        if key in env_section:
            kwargs[key] = env_section[key]

    return kwargs, map_name


def load_training_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    return raw["training"]


def main(config_path: Path | None = None):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    ppo_config = load_ppo_config(config_path)
    env_kwargs, map_name = load_env_config(config_path)
    training = load_training_config(config_path)

    artifacts_dir = PROJECT_ROOT / training["artifacts_dir"]
    device = torch.device("cpu")

    trainer = Trainer(ppo_config, env_kwargs, artifacts_dir, map_name, device)
    trainer.train()


if __name__ == "__main__":
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(config_path)
