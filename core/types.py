from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core.evaluator import MapEvalResult


# ── Request objects ─────────────────────────────────────

@dataclass
class TrainRequest:
    """Everything needed to start a training run."""
    algo_config: SimpleNamespace
    env_config: dict[str, Any]
    curriculum: dict[str, Any]
    training_config: dict[str, Any]
    general_config: dict[str, Any]
    dashboard_config: dict[str, Any]
    metrics_config: dict[str, Any]
    config_path: Path
    artifacts_root: Path
    algo_name: str = "ppo"
    resume_from: Path | None = None
    run_id: str | None = None
    load_weights_from: Path | None = None


@dataclass
class EvalRequest:
    """Everything needed to run an evaluation."""
    map_ids: list[int]
    num_episodes: int
    npc_count: int | None
    station_count: int | None
    run_id: str | None
    step: int | None
    gif_fps: int
    output_dir: Path | None
    algo_config: SimpleNamespace
    env_config: dict[str, Any]
    artifacts_root: Path


# ── Context objects ─────────────────────────────────────

@dataclass
class RunContext:
    """Resolved runtime paths and metadata for a training run."""
    artifacts_root: Path
    checkpoints_root: Path
    run_dir: Path
    checkpoint_dir: Path
    train_log_path: Path
    run_id: str
    seed: int


@dataclass
class EvalContext:
    """Resolved paths and metadata for an evaluation run."""
    checkpoints_root: Path
    run_dir: Path
    model_path: Path
    checkpoint_step: int
    eval_dir: Path
    map_configs: list[dict[str, Any]]
    map_names: list[str]


# ── Result objects ──────────────────────────────────────

@dataclass
class TrainResult:
    run_id: str
    run_dir: Path
    total_steps: int
    success: bool = True
    error: str | None = None


@dataclass
class EvalResult:
    checkpoints_root: Path
    run_dir: Path
    model_path: Path
    checkpoint_step: int
    eval_dir: Path
    results: list[MapEvalResult] = field(default_factory=list)
    overall_avg_reward: float = 0.0
    overall_avg_score: float = 0.0
    overall_avg_steps: float = 0.0
    overall_avg_charges: float = 0.0
    total_episodes: int = 0
    success: bool = True
    error: str | None = None
