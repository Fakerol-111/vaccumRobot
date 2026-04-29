"""入口模块：读取配置文件，组装 Trainer 并启动训练。

通过 `python train_sync.py` 或 `python main.py` 直接运行。
支持多地图轮换 + 课程学习。
"""

from __future__ import annotations

import argparse
import random
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from config.map_loader import load_map_configs
from trainer import Trainer, _find_nearest_checkpoint

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


def _load_general_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    general = raw.get("general", {})
    return {
        "seed": int(general.get("seed", 42)),
    }


def _load_dashboard_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = _default_config_path()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    dashboard = raw.get("dashboard", {})
    return {
        "enabled": bool(dashboard.get("enabled", False)),
        "host": dashboard.get("host", "0.0.0.0"),
        "port": int(dashboard.get("port", 8088)),
    }


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for Robot Vacuum")
    parser.add_argument("config", nargs="?", type=str, default=None,
                        help="Path to config TOML file")
    parser.add_argument("--resume", nargs="?", const="auto", type=str, default=None,
                        help="Resume from latest checkpoint in the run directory. "
                             "Use --resume <checkpoint_path> for a specific checkpoint, "
                             "or --resume (without value) to auto-detect.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Alias for --resume <path>")
    return parser.parse_args(argv)


def main(config_path: Path | None = None, resume_from: Path | str | None = None):
    general = _load_general_config(config_path)
    seed = general["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    ppo_config = load_ppo_config(config_path)
    env_settings = load_env_config(config_path)
    curriculum = load_curriculum(config_path)
    training = load_training_config(config_path)

    if resume_from is None:
        cfg_resume = training.get("resume_from", "")
        if cfg_resume:
            resume_from = cfg_resume

    artifacts_dir = PROJECT_ROOT / training["artifacts_dir"]
    device = torch.device("cpu")

    default_npc = env_settings["default_npc_count"]
    default_station = env_settings["default_station_count"]
    map_strategy = env_settings["map_strategy"]
    default_map_list = env_settings["default_map_list"]

    collector = None
    dashboard_server = None
    dashboard_config = _load_dashboard_config(config_path)
    if dashboard_config["enabled"]:
        from training_dashboard import MetricsCollector, DashboardServer
        collector = MetricsCollector()
        dashboard_server = DashboardServer(
            collector,
            host=dashboard_config["host"],
            port=dashboard_config["port"],
        )
        dashboard_server.start()

    resolved_resume: Path | None = None
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.is_dir():
            found = _find_nearest_checkpoint(resume_path)
            if found is None:
                print(f"[train_sync] No checkpoint found in {resume_path}")
                resume_path = None
            else:
                resume_path = found
        if resume_path is not None and not resume_path.exists():
            print(f"[train_sync] Checkpoint not found: {resume_path}")
            resume_path = None
        resolved_resume = resume_path

    trainer = Trainer(
        ppo_config=ppo_config,
        default_npc_count=default_npc,
        default_station_count=default_station,
        map_strategy=map_strategy,
        curriculum=curriculum,
        artifacts_dir=artifacts_dir,
        device=device,
        collector=collector,
        default_map_list=default_map_list,
        seed=seed,
        resume_from=resolved_resume,
        config_path=config_path,
    )
    try:
        trainer.train()
    finally:
        if dashboard_server is not None:
            dashboard_server.stop()


if __name__ == "__main__":
    args = _parse_args()
    config_path = Path(args.config) if args.config else None

    resume_path: str | Path | None = None
    if args.checkpoint:
        resume_path = args.checkpoint
    elif args.resume:
        if args.resume == "auto" and config_path is not None:
            training = load_training_config(config_path)
            artifacts_dir = PROJECT_ROOT / training["artifacts_dir"]
            map_dir = artifacts_dir / "multi_map" / "checkpoints"
            if map_dir.exists():
                run_dirs = sorted(map_dir.iterdir(), reverse=True)
                if run_dirs:
                    found = _find_nearest_checkpoint(run_dirs[0])
                    if found:
                        resume_path = found
                        print(f"[train_sync] Auto-resume: found {found}")
                    else:
                        print(f"[train_sync] Auto-resume: no checkpoint in {run_dirs[0]}")
                else:
                    print("[train_sync] Auto-resume: no run directories found")
            else:
                print(f"[train_sync] Auto-resume: {map_dir} does not exist")
        else:
            resume_path = args.resume

    main(config_path, resume_from=resume_path)
