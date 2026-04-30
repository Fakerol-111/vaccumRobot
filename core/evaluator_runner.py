from __future__ import annotations

import logging
from pathlib import Path

from agent.ppo import PPOAlgorithm

logger = logging.getLogger(__name__)
from agent.preprocessor import Preprocessor
from configs.runtime_config import build_multi_env_configs
from core import get_device
from core.paths import find_checkpoint, find_run_dir, get_artifacts_root, get_checkpoints_root, get_eval_dir
from core.types import EvalContext, EvalRequest, EvalResult
from core.evaluator import evaluate_multi_map_with_recording


def _resolve_eval_context(req: EvalRequest) -> EvalContext:
    artifacts_root = get_artifacts_root(req.artifacts_root)
    checkpoints_root = get_checkpoints_root(artifacts_root)

    run_dir = find_run_dir(checkpoints_root, req.run_id)
    if run_dir is None:
        raise FileNotFoundError(f"No checkpoint runs found in {checkpoints_root}")

    model_path = find_checkpoint(run_dir, req.step)
    if model_path is None:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")

    checkpoint_step = int(model_path.stem.split("_")[1])

    npc_count = req.npc_count
    if npc_count is None:
        npc_count = req.env_config.get("default_npc_count", 1)

    station_count = req.station_count
    if station_count is None:
        station_count = req.env_config.get("default_station_count", 4)

    map_configs = build_multi_env_configs(req.map_ids, npc_count, station_count)
    map_names = [f"map_{mid}" for mid in req.map_ids]

    eval_dir = get_eval_dir(run_dir, checkpoint_step, req.output_dir)

    return EvalContext(
        checkpoints_root=checkpoints_root,
        run_dir=run_dir,
        model_path=model_path,
        checkpoint_step=checkpoint_step,
        eval_dir=eval_dir,
        map_configs=map_configs,
        map_names=map_names,
    )


def _print_results(result: EvalResult) -> None:
    print()
    print("=== Per-Map Results ===")
    header = (
        f"{'map':<12} {'ep':>4} {'avg_reward':>11} "
        f"{'avg_score':>10} {'avg_steps':>10} {'avg_charges':>12}"
    )
    print(header)
    print("-" * len(header))
    for res in result.results:
        print(
            f"{res.map_name:<12} {res.num_episodes:>4} "
            f"{res.avg_reward:>11.2f} {res.avg_score:>10.1f} "
            f"{res.avg_steps:>10.1f} {res.avg_charges:>12.2f}"
        )
    print("-" * len(header))
    print(
        f"{'OVERALL':<12} {result.total_episodes:>4} "
        f"{result.overall_avg_reward:>11.2f} {result.overall_avg_score:>10.1f} "
        f"{result.overall_avg_steps:>10.1f} {result.overall_avg_charges:>12.2f}"
    )
    print()
    logger.info("Files saved to %s", result.eval_dir)
    for res in result.results:
        logger.info("  %s/summary.txt", res.map_name)
        logger.info("  %s/*.gif", res.map_name)


def run_evaluation(req: EvalRequest) -> EvalResult:
    try:
        ctx = _resolve_eval_context(req)
    except FileNotFoundError as e:
        return EvalResult(
            checkpoints_root=get_checkpoints_root(get_artifacts_root(req.artifacts_root)),
            run_dir=Path(),
            model_path=Path(),
            checkpoint_step=0,
            eval_dir=Path(),
            success=False,
            error=str(e),
        )

    logger.info("Run:          %s", ctx.run_dir.name)
    logger.info("Step:         %s", ctx.checkpoint_step)
    logger.info("Model:        %s", ctx.model_path)
    logger.info("Maps:         %s", ctx.map_names)
    logger.info("Episodes/map: %s", req.num_episodes)
    logger.info("GIF FPS:      %s", req.gif_fps)
    logger.info("Output:       %s", ctx.eval_dir)

    device = get_device()
    algorithm = PPOAlgorithm(req.algo_config, device)
    algorithm.load(ctx.model_path)
    preprocessor = Preprocessor()

    raw = evaluate_multi_map_with_recording(
        algorithm,
        preprocessor,
        ctx.map_configs,
        ctx.map_names,
        ctx.eval_dir,
        num_episodes=req.num_episodes,
        gif_fps=req.gif_fps,
    )

    result = EvalResult(
        checkpoints_root=ctx.checkpoints_root,
        run_dir=ctx.run_dir,
        model_path=ctx.model_path,
        checkpoint_step=ctx.checkpoint_step,
        eval_dir=ctx.eval_dir,
        results=raw["results"],
        overall_avg_reward=raw["overall_avg_reward"],
        overall_avg_score=raw["overall_avg_score"],
        overall_avg_steps=raw["overall_avg_steps"],
        overall_avg_charges=raw["overall_avg_charges"],
        total_episodes=raw["total_episodes"],
    )

    _print_results(result)
    return result
