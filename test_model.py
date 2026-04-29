"""测试模块：加载训练断点，在多个地图上评估并录制所有 episode 的轨迹 GIF。

所有评估参数在 config/test_config.toml 中配置。
GIF 按地图分目录存放，输出每个地图的统计结果及全局汇总。

用法：
    python test_model.py                              # 使用默认 config/test_config.toml
    python test_model.py --config config/other.toml   # 指定自定义测试配置
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import torch

from agent.agent import Agent

PROJECT_ROOT = Path(__file__).resolve().parent


def load_test_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "test_config.toml"
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    test = raw.get("test", {})

    maps = list(test.get("maps", [1, 2, 3, 4]))
    episodes = int(test.get("episodes", 10))
    npc_count = test.get("npc_count")
    station_count = test.get("station_count")
    run_id = test.get("run_id", "")
    step = int(test.get("step", 0))
    gif_fps = int(test.get("gif_fps", 10))
    output_dir = test.get("output_dir", "")

    if run_id == "":
        run_id = None
    if step == 0:
        step = None

    return {
        "maps": maps,
        "episodes": episodes,
        "npc_count": npc_count,
        "station_count": station_count,
        "run_id": run_id,
        "step": step,
        "gif_fps": gif_fps,
        "output_dir": output_dir if output_dir else None,
    }


def _find_run_dir(checkpoints_root: Path, run_id: str | None) -> Path | None:
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
    config_path: Path | None = None

    args = list(sys.argv[1:])
    i = 0
    while i < len(args):
        if args[i] == "--config" and i + 1 < len(args):
            config_path = Path(args[i + 1])
            i += 2
        else:
            print(f"Usage: python test_model.py [--config <path>]")
            print(f"  --config  指定测试配置文件（默认 config/test_config.toml）")
            sys.exit(1)

    test_cfg = load_test_config(config_path)

    from train_sync import load_ppo_config, load_env_config, build_multi_env_configs

    ppo_config = load_ppo_config()
    env_settings = load_env_config()

    artifacts_dir = PROJECT_ROOT / "artifacts"
    checkpoints_root = artifacts_dir / "multi_map" / "checkpoints"

    run_id = test_cfg["run_id"]
    step = test_cfg["step"]

    run_dir = _find_run_dir(checkpoints_root, run_id)
    if run_dir is None:
        print(f"No checkpoint runs found in {checkpoints_root}")
        sys.exit(1)

    model_path = _find_checkpoint(run_dir, step)
    if model_path is None:
        print(f"No checkpoint found in {run_dir}")
        sys.exit(1)

    checkpoint_step = int(model_path.stem.split("_")[1])

    map_ids = test_cfg["maps"]
    num_episodes = test_cfg["episodes"]

    npc_count = test_cfg["npc_count"]
    if npc_count is None:
        npc_count = env_settings.get("default_npc_count", 1)

    station_count = test_cfg["station_count"]
    if station_count is None:
        station_count = env_settings.get("default_station_count", 4)

    if test_cfg["output_dir"]:
        eval_dir = Path(test_cfg["output_dir"])
    else:
        eval_dir = run_dir / f"eval_{checkpoint_step}"

    gif_fps = test_cfg["gif_fps"]

    print(f"Run:          {run_dir.name}")
    print(f"Step:         {checkpoint_step}")
    print(f"Model:        {model_path}")
    print(f"Maps:         {[f'map_{mid}' for mid in map_ids]}")
    print(f"Episodes/map: {num_episodes}")
    print(f"NPCs:         {npc_count}  Stations: {station_count}")
    print(f"GIF FPS:      {gif_fps}")
    print(f"Output:       {eval_dir}")

    map_configs = build_multi_env_configs(map_ids, npc_count, station_count)
    map_names = [f"map_{mid}" for mid in map_ids]

    device = torch.device("cpu")
    agent = Agent(ppo_config, device)
    agent.load(model_path)

    from evaluator import evaluate_multi_map_with_recording

    result = evaluate_multi_map_with_recording(
        agent, map_configs, map_names, eval_dir, num_episodes=num_episodes, gif_fps=gif_fps,
    )

    print()
    print("=== Per-Map Results ===")
    header = f"{'map':<12} {'ep':>4} {'avg_reward':>11} {'avg_score':>10} {'avg_steps':>10} {'avg_charges':>12}"
    print(header)
    print("-" * len(header))
    for res in result["results"]:
        print(
            f"{res.map_name:<12} {res.num_episodes:>4} "
            f"{res.avg_reward:>11.2f} {res.avg_score:>10.1f} "
            f"{res.avg_steps:>10.1f} {res.avg_charges:>12.2f}"
        )

    print("-" * len(header))
    print(
        f"{'OVERALL':<12} {result['total_episodes']:>4} "
        f"{result['overall_avg_reward']:>11.2f} {result['overall_avg_score']:>10.1f} "
        f"{result['overall_avg_steps']:>10.1f} {result['overall_avg_charges']:>12.2f}"
    )
    print()
    print(f"Files saved to {eval_dir}")
    print(f"  summary_all.txt  - overall summary")
    for mn in map_names:
        print(f"  {mn}/summary.txt  - per-map summary")
        print(f"  {mn}/*.gif        - trajectory GIFs")


if __name__ == "__main__":
    main()
