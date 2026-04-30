from __future__ import annotations

import logging
import random
from datetime import datetime

logger = logging.getLogger(__name__)

import numpy as np
import torch

from agent.preprocessor import Preprocessor
from agent.registry import get as get_algorithm
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

    # --load-weights 与 resume 互斥
    if req.load_weights_from is not None:
        resume_from = None
    else:
        resume_from = _resolve_resume(req.resume_from, req.training_config)

    run_id = req.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_resume = resolve_checkpoint(resume_from, req.artifacts_root)

    run_dir = get_run_dir(checkpoints_root, run_id)

    env = req.env_config
    collector, dashboard_server = create_dashboard(req.dashboard_config)

    device = get_device()
    algo_cls = get_algorithm(req.algo_name)
    algorithm = algo_cls(req.algo_config, device)

    # 只加载权重（不计步、不恢复优化器状态），从 step 0 开始训练
    if req.load_weights_from is not None:
        algorithm.load(str(req.load_weights_from))
        logger.info("Loaded weights from %s, starting training from step 0", req.load_weights_from)

    # 注入 dashboard collector 到算法指标采集器
    if collector is not None:
        reporter = algorithm.metrics_reporter
        if reporter is not None:
            reporter.set_collector(collector)

    preprocessor = Preprocessor()

    trainer = Trainer(
        algorithm=algorithm,
        preprocessor=preprocessor,
        algo_config=req.algo_config,
        default_npc_count=env["default_npc_count"],
        default_station_count=env["default_station_count"],
        map_strategy=env["map_strategy"],
        curriculum=req.curriculum,
        artifacts_dir=req.artifacts_root,
        device=device,
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
        total_steps=req.algo_config.total_timesteps,
    )
