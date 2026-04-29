from __future__ import annotations

import random
from datetime import datetime

import numpy as np
import torch

from core import get_device
from core.paths import get_checkpoints_root, get_run_dir
from core.types import TrainRequest, TrainResult
from services.checkpoint_service import resolve_checkpoint
from services.dashboard_service import create_dashboard
from core.trainer import Trainer


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_resume(
    resume_from: str | None,
    training_config: dict,
) -> str | None:
    if resume_from is None:
        cfg_resume = training_config.get("resume_from", "")
        if cfg_resume:
            resume_from = cfg_resume
    return resume_from


def run_training(req: TrainRequest) -> TrainResult:
    _set_seeds(req.general_config["seed"])

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)

    checkpoints_root = get_checkpoints_root(req.artifacts_root)
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    resume_from = _resolve_resume(req.resume_from, req.training_config)

    resolved_resume = resolve_checkpoint(resume_from, req.artifacts_root)
    if resolved_resume is not None and req.run_id is None:
        run_id = resolved_resume.parent.name
    else:
        run_id = req.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = get_run_dir(checkpoints_root, run_id)

    env = req.env_config
    collector, dashboard_server = create_dashboard(req.dashboard_config)

    trainer = Trainer(
        ppo_config=req.ppo_config,
        default_npc_count=env["default_npc_count"],
        default_station_count=env["default_station_count"],
        map_strategy=env["map_strategy"],
        curriculum=req.curriculum,
        artifacts_dir=req.artifacts_root,
        device=get_device(),
        collector=collector,
        default_map_list=env["default_map_list"],
        seed=req.general_config["seed"],
        resume_from=resolved_resume,
        run_id=run_id,
        config_path=req.config_path,
        metrics_config=req.metrics_config,
    )

    try:
        trainer.train()
    finally:
        if dashboard_server is not None:
            dashboard_server.stop()

    return TrainResult(
        run_id=run_id,
        run_dir=run_dir,
        total_steps=req.ppo_config.total_timesteps,
    )
