"""测试模块：加载训练断点，运行评估并录制所有 episode 的轨迹。

用法：
    python test_model.py                                    # 自动找最新 run 的最新 checkpoint
    python test_model.py --step 4000                        # 最新 run 的 checkpoint_4000.pt
    python test_model.py --run 20260428_143052              # 指定 run 的最新 checkpoint
    python test_model.py --run 20260428_143052 --step 4000  # 指定 run 和步数
    python test_model.py --config config/other.toml         # 指定配置文件
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from agent.agent import Agent

PROJECT_ROOT = Path(__file__).resolve().parent


def _find_run_dir(map_dir: Path, run_id: str | None) -> Path | None:
    checkpoints_root = map_dir / "checkpoints"
    if not checkpoints_root.is_dir():
        return None

    if run_id:
        run_dir = checkpoints_root / run_id
        return run_dir if run_dir.is_dir() else None

    runs = sorted(d for d in checkpoints_root.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def _find_checkpoint(run_dir: Path, step: int | None) -> Path | None:
    if step is not None:
        path = run_dir / f"checkpoint_{step}.pt"
        return path if path.exists() else None

    checkpoints = sorted(
        run_dir.glob("checkpoint_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    return checkpoints[-1] if checkpoints else None


def main():
    from train_sync import load_ppo_config, load_env_config, load_training_config

    config_path: Path | None = None
    run_id: str | None = None
    step: int | None = None

    args = list(sys.argv[1:])
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = Path(args[i + 1])
            i += 2
        elif args[i] == "--run" and i + 1 < len(args):
            run_id = args[i + 1]
            i += 2
        elif args[i] == "--step" and i + 1 < len(args):
            step = int(args[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)

    ppo_config = load_ppo_config(config_path)
    env_kwargs, map_name = load_env_config(config_path)
    training = load_training_config(config_path)

    artifacts_dir = PROJECT_ROOT / training["artifacts_dir"]
    map_dir = artifacts_dir / map_name

    run_dir = _find_run_dir(map_dir, run_id)
    if run_dir is None:
        print(f"No checkpoint runs found in {map_dir / 'checkpoints'}")
        sys.exit(1)

    model_path = _find_checkpoint(run_dir, step)
    if model_path is None:
        print(f"No checkpoint found in {run_dir}")
        sys.exit(1)

    checkpoint_step = int(model_path.stem.split("_")[1])

    eval_dir = run_dir / f"eval_{checkpoint_step}"

    print(f"Run:     {run_dir.name}")
    print(f"Map:     {map_name}")
    print(f"Step:    {checkpoint_step}")
    print(f"Model:   {model_path}")
    print(f"Output:  {eval_dir}")

    device = torch.device("cpu")
    agent = Agent(ppo_config, device)
    agent.load(model_path)

    from evaluator import evaluate_with_recording

    num_episodes = int(training.get("eval_episodes", 10))
    result = evaluate_with_recording(agent, env_kwargs, eval_dir, num_episodes=num_episodes)
    print(
        f"Avg reward: {result['avg_reward']:.1f}  "
        f"Avg steps: {result['avg_steps']:.0f}  "
        f"Range: [{result['min_reward']:.0f}, {result['max_reward']:.0f}]"
    )
    print(f"Files saved to {eval_dir}")


if __name__ == "__main__":
    main()
