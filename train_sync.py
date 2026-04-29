"""入口模块：读取配置文件，组装 Trainer 并启动训练。

通过 `python train_sync.py` 或 `python main.py` 直接运行。
支持多地图轮换 + 课程学习。
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from config.map_loader import load_map_configs
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
    return SimpleNamespace(
        num_actors=int(ppo.get("num_actors", 4)),
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
        max_npcs=int(ppo.get("max_npcs", 5)),
        local_view_size=int(ppo.get("local_view_size", 21)),
        num_actions=int(ppo.get("num_actions", 8)),
    )


def load_env_config(config_path: Path | None = None) -> dict[str, Any]:
    """加载 [env] 节的默认参数（npc_count, station_count, map_strategy 等）。"""
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    env = raw.get("env", {})
    return {
        "default_map_list": env.get("default_map_list", [1]),
        "default_npc_count": int(env.get("default_npc_count", 1)),
        "default_station_count": int(env.get("default_station_count", 4)),
        "map_strategy": env.get("map_strategy", "round_robin"),
    }


def load_curriculum(config_path: Path | None = None) -> dict[str, Any]:
    """加载课程学习配置。

    Returns:
        dict with keys:
            enabled: bool
            stages: list[dict] | None  每个 stage 包含 {name, maps, npc_count, station_count, total_steps}
    """
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    curriculum = raw.get("curriculum", {})
    enabled = bool(curriculum.get("enabled", False))
    stages_raw = curriculum.get("stage", [])

    stages = []
    cumulative = 0
    for s in stages_raw:
        name = s.get("name", f"stage_{len(stages)}")
        maps = list(s.get("maps", [1]))
        npc_count = int(s.get("npc_count", 1))
        station_count = int(s.get("station_count", 4))
        stage_steps = int(s.get("total_steps", 0))
        cumulative += stage_steps
        stages.append({
            "name": name,
            "maps": maps,
            "npc_count": npc_count,
            "station_count": station_count,
            "total_steps": cumulative,
        })

    return {"enabled": enabled, "stages": stages}


def load_training_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    return raw["training"]


def build_multi_env_configs(
    map_ids: list[int],
    npc_count: int,
    station_count: int,
) -> list[dict[str, Any]]:
    """根据 map ID 列表创建完整的环境配置列表。"""
    base_configs = load_map_configs(map_ids)
    env_configs = []
    for cfg in base_configs:
        cfg = dict(cfg)
        if "npc_count" not in cfg:
            cfg["npc_count"] = npc_count
        if "station_count" not in cfg:
            cfg["station_count"] = station_count
        env_configs.append(cfg)
    return env_configs


def main(config_path: Path | None = None):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    ppo_config = load_ppo_config(config_path)
    env_settings = load_env_config(config_path)
    curriculum = load_curriculum(config_path)
    training = load_training_config(config_path)

    artifacts_dir = PROJECT_ROOT / training["artifacts_dir"]
    device = torch.device("cpu")

    default_npc = env_settings["default_npc_count"]
    default_station = env_settings["default_station_count"]
    map_strategy = env_settings["map_strategy"]

    trainer = Trainer(
        ppo_config=ppo_config,
        default_npc_count=default_npc,
        default_station_count=default_station,
        map_strategy=map_strategy,
        curriculum=curriculum,
        artifacts_dir=artifacts_dir,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    main(config_path)
