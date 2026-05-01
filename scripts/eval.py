"""评估入口：加载训练断点，在多个地图上评估并录制所有 episode 的轨迹 GIF。

用法：
    python scripts/eval.py [--config configs/test_config.toml]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.runtime_config import load_test_config_bundle, load_train_config_bundle
from core.evaluator_runner import run_evaluation
from core.types import EvalRequest


def parse_args(argv: list[str] | None = None) -> Path | None:
    config_path: Path | None = None
    args = list(argv if argv is not None else sys.argv[1:])
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = Path(args[i + 1])
            i += 2
        else:
            print(f"Usage: python scripts/eval.py [--config <path>]")
            print(f"  --config  指定测试配置文件（默认 configs/test_config.toml）")
            sys.exit(1)
    return config_path


def main(config_path: Path | None = None):
    test_cfg = load_test_config_bundle(config_path)
    train_cfg = load_train_config_bundle()

    algo_name = train_cfg.algo["name"]
    algo_config = getattr(train_cfg, algo_name, None)
    if algo_config is not None and algo_name == "trpo" and algo_config.model_type != "separate":
        raise ValueError(
            "TRPO 必须使用分离的 actor-critic 网络，请在配置中设置 "
            "[algorithm]\n  model_type = \"separate\""
        )

    req = EvalRequest(
        map_ids=test_cfg["maps"],
        num_episodes=test_cfg["episodes"],
        npc_count=test_cfg["npc_count"],
        station_count=test_cfg["station_count"],
        run_id=test_cfg["run_id"],
        step=test_cfg["step"],
        gif_fps=test_cfg["gif_fps"],
        output_dir=Path(test_cfg["output_dir"]) if test_cfg["output_dir"] else None,
        algo_config=algo_config or train_cfg.ppo,
        algo_name=algo_name,
        env_config=train_cfg.env,
        artifacts_root=PROJECT_ROOT / train_cfg.training["artifacts_dir"],
    )

    result = run_evaluation(req)
    if not result.success:
        logger.error("Evaluation failed: %s", result.error)
        sys.exit(1)


if __name__ == "__main__":
    from core import setup_logging
    setup_logging()

    config_path = parse_args()
    main(config_path)
