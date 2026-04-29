"""训练入口：解析参数，组装请求，启动训练。

用法：
    python scripts/train.py [config_path] [--resume [auto|<path>]] [--checkpoint <path>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.runtime_config import load_train_config_bundle
from core.trainer_runner import run_training
from core.types import TrainRequest
from services.checkpoint_service import resolve_auto_resume


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    cfg = load_train_config_bundle(config_path)

    req = TrainRequest(
        ppo_config=cfg.ppo,
        env_config=cfg.env,
        curriculum=cfg.curriculum,
        training_config=cfg.training,
        general_config=cfg.general,
        dashboard_config=cfg.dashboard,
        metrics_config=cfg.metrics,
        config_path=cfg.config_path,
        artifacts_root=PROJECT_ROOT / cfg.training["artifacts_dir"],
        resume_from=Path(resume_from) if resume_from else None,
    )

    result = run_training(req)
    if result.error:
        print(f"[train] Training failed: {result.error}")


if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config) if args.config else None

    resume_path: str | Path | None = None
    if args.checkpoint:
        resume_path = args.checkpoint
    elif args.resume:
        if args.resume == "auto" and config_path is not None:
            cfg = load_train_config_bundle(config_path)
            artifacts_root = PROJECT_ROOT / cfg.training["artifacts_dir"]
            resume_path = resolve_auto_resume(artifacts_root)
        else:
            resume_path = args.resume

    main(config_path, resume_from=resume_path)
