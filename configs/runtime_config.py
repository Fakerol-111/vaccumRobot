from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from configs.map_loader import load_map_configs

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ── 默认路径 ───────────────────────────────────────────

def get_default_train_config_path() -> Path:
    return _PROJECT_ROOT / "configs" / "train_config.toml"


def get_default_test_config_path() -> Path:
    return _PROJECT_ROOT / "configs" / "test_config.toml"


# ── 底层 TOML 读取 ─────────────────────────────────────

def _load_toml(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = get_default_train_config_path()
    with open(path, "rb") as f:
        return tomllib.load(f)


# ── 各节解析（内部） ───────────────────────────────────

def _parse_ppo(raw: dict[str, Any]) -> SimpleNamespace:
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
        save_time_interval=float(ppo.get("save_time_interval", 0)),
        log_interval=int(ppo.get("log_interval", 500)),
        max_npcs=int(ppo.get("max_npcs", 5)),
        local_view_size=int(ppo.get("local_view_size", 21)),
        num_actions=int(ppo.get("num_actions", 8)),
    )


def _parse_env(raw: dict[str, Any]) -> dict[str, Any]:
    env = raw.get("env", {})
    return {
        "default_map_list": env.get("default_map_list", [1]),
        "default_npc_count": int(env.get("default_npc_count", 1)),
        "default_station_count": int(env.get("default_station_count", 4)),
        "map_strategy": env.get("map_strategy", "round_robin"),
    }


def _parse_curriculum(raw: dict[str, Any]) -> dict[str, Any]:
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


def _parse_training(raw: dict[str, Any]) -> dict[str, Any]:
    return raw["training"]


def _parse_general(raw: dict[str, Any]) -> dict[str, Any]:
    general = raw.get("general", {})
    return {"seed": int(general.get("seed", 42))}


def _parse_dashboard(raw: dict[str, Any]) -> dict[str, Any]:
    dashboard = raw.get("dashboard", {})
    return {
        "enabled": bool(dashboard.get("enabled", False)),
        "host": dashboard.get("host", "0.0.0.0"),
        "port": int(dashboard.get("port", 8088)),
    }


def _parse_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    metrics = raw.get("metrics", {})
    return {
        "max_updates": int(metrics.get("max_updates", 500)),
        "max_episodes": int(metrics.get("max_episodes", 500)),
    }


def _parse_algorithm(raw: dict[str, Any]) -> dict[str, Any]:
    algo = raw.get("algorithm", {})
    return {
        "name": algo.get("name", "ppo"),
    }


def _parse_grpo(raw: dict[str, Any]) -> SimpleNamespace:
    grpo = raw.get("grpo", {})
    return SimpleNamespace(
        learning_rate=float(grpo.get("learning_rate", 3e-4)),
        gamma=float(grpo.get("gamma", 0.99)),
        max_grad_norm=float(grpo.get("max_grad_norm", 0.5)),
        total_timesteps=int(grpo.get("total_timesteps", 10_000)),
        save_interval=int(grpo.get("save_interval", 5_000)),
        save_time_interval=float(grpo.get("save_time_interval", 0)),
        log_interval=int(grpo.get("log_interval", 500)),
        max_npcs=int(grpo.get("max_npcs", 5)),
        local_view_size=int(grpo.get("local_view_size", 21)),
        num_actions=int(grpo.get("num_actions", 8)),
        batch_size=int(grpo.get("batch_size", 256)),
        branch_window=int(grpo.get("branch_window", 200)),
        branch_interval=int(grpo.get("branch_interval", 30)),
        num_candidates=int(grpo.get("num_candidates", 4)),
        kl_coef=float(grpo.get("kl_coef", 0.1)),
        action_prob_threshold=float(grpo.get("action_prob_threshold", 0.01)),
    )


# ── 公开接口 ───────────────────────────────────────────

def load_train_config_bundle(path: Path | None = None) -> SimpleNamespace:
    """加载训练配置（train_config.toml），返回包含所有配置节的 bundle。

    Returns:
        SimpleNamespace 包含以下属性:
            algo        – dict {name} 算法名称
            ppo         – SimpleNamespace (PPO 超参)
            grpo        – SimpleNamespace (GRPO 超参)
            env         – dict (环境默认参数)
            curriculum  – dict {enabled, stages}
            training    – dict (训练产物路径等)
            general     – dict {seed}
            dashboard   – dict {enabled, host, port}
            metrics     – dict {max_updates, max_episodes}
            config_path – Path (实际读入的配置文件路径)
    """
    resolved = path if path is not None else get_default_train_config_path()
    raw = _load_toml(resolved)

    return SimpleNamespace(
        algo=_parse_algorithm(raw),
        ppo=_parse_ppo(raw),
        grpo=_parse_grpo(raw),
        env=_parse_env(raw),
        curriculum=_parse_curriculum(raw),
        training=_parse_training(raw),
        general=_parse_general(raw),
        dashboard=_parse_dashboard(raw),
        metrics=_parse_metrics(raw),
        config_path=resolved,
    )


def load_test_config_bundle(path: Path | None = None) -> dict[str, Any]:
    """加载测试配置（test_config.toml），返回 dict。

    Returns:
        dict with keys: maps, episodes, npc_count, station_count,
                        run_id, step, gif_fps, output_dir
    """
    resolved = path if path is not None else get_default_test_config_path()
    raw = _load_toml(resolved)
    test = raw.get("test", {})

    maps = list(test.get("maps", [1, 2, 3, 4]))
    episodes = int(test.get("episodes", 10))
    npc_count = test.get("npc_count")
    station_count = test.get("station_count")
    run_id = test.get("run_id", "") or None
    step = int(test.get("step", 0)) or None
    gif_fps = int(test.get("gif_fps", 10))
    output_dir = test.get("output_dir", "") or None

    return {
        "maps": maps,
        "episodes": episodes,
        "npc_count": npc_count,
        "station_count": station_count,
        "run_id": run_id,
        "step": step,
        "gif_fps": gif_fps,
        "output_dir": output_dir,
    }


# ── 兼容性导出（保持原有函数名可用） ─────────────────

def load_ppo_config(config_path: Path | None = None) -> SimpleNamespace:
    return load_train_config_bundle(config_path).ppo


def load_env_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_train_config_bundle(config_path).env


def load_curriculum(config_path: Path | None = None) -> dict[str, Any]:
    return load_train_config_bundle(config_path).curriculum


def load_training_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_train_config_bundle(config_path).training


def load_test_config(config_path: Path | None = None) -> dict[str, Any]:
    return load_test_config_bundle(config_path)


# ── 工具函数 ───────────────────────────────────────────

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
