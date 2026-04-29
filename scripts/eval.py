"""评估入口：加载训练断点，在多个地图上评估并录制所有 episode 的轨迹 GIF。

用法：
    python scripts/eval.py [--config configs/test_config.toml]
"""

from __future__ import annotations

import sys
from pathlib import Path

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

    req = EvalRequest(
        map_ids=test_cfg["maps"],
        num_episodes=test_cfg["episodes"],
        npc_count=test_cfg["npc_count"],
        station_count=test_cfg["station_count"],
        run_id=test_cfg["run_id"],
        step=test_cfg["step"],
        gif_fps=test_cfg["gif_fps"],
        output_dir=Path(test_cfg["output_dir"]) if test_cfg["output_dir"] else None,
        ppo_config=train_cfg.ppo,
        env_config=train_cfg.env,
        artifacts_root=PROJECT_ROOT / "artifacts",
    )

    result = run_evaluation(req)
    if not result.success:
        print(f"[eval] Evaluation failed: {result.error}")
        sys.exit(1)


if __name__ == "__main__":
    config_path = parse_args()
    main(config_path)
