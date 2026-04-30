from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agent.base import Algorithm
from agent.preprocessor import Preprocessor
from env import TrajectoryRecorder
from env.factory import create_env


@dataclass
class MapEvalResult:
    map_name: str
    rewards: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    charges: list[int] = field(default_factory=list)

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def avg_steps(self) -> float:
        return float(np.mean(self.steps)) if self.steps else 0.0

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0

    @property
    def avg_charges(self) -> float:
        return float(np.mean(self.charges)) if self.charges else 0.0

    @property
    def num_episodes(self) -> int:
        return len(self.rewards)


def evaluate(
    algorithm: Algorithm,
    preprocessor: Preprocessor,
    env_kwargs: dict[str, Any],
    num_episodes: int = 3,
) -> dict:
    eval_env = create_env(env_kwargs, enable_recording=False, render_mode=None)

    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    for _ in range(num_episodes):
        preprocessor.reset()
        payload = eval_env.reset(options={"mode": "eval"})
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            map_img, vector, legal_action, reward = preprocessor.feature_process(
                payload, preprocessor.curr_action
            )
            act_result = algorithm.act(map_img, vector, np.asarray(legal_action, dtype=np.float32), mode="exploit")
            payload = eval_env.step(act_result.action)
            total_reward += reward
            steps += 1
            done = payload["terminated"] or payload["truncated"]

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    eval_env.close()

    return _build_result(episode_rewards, episode_steps)


def evaluate_with_recording(
    algorithm: Algorithm,
    preprocessor: Preprocessor,
    env_kwargs: dict[str, Any],
    eval_dir: Path,
    num_episodes: int = 10,
) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_scores: list[int] = []
    episode_charges: list[int] = []

    for ep in range(num_episodes):
        preprocessor.reset()
        recorder = TrajectoryRecorder()
        eval_env = create_env(
            env_kwargs,
            enable_recording=True,
            render_mode=None,
            trajectory_recorder=recorder,
        )
        payload = eval_env.reset(options={"mode": "eval"})
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            map_img, vector, legal_action, reward = preprocessor.feature_process(
                payload, preprocessor.curr_action
            )
            act_result = algorithm.act(map_img, vector, np.asarray(legal_action, dtype=np.float32), mode="exploit")
            payload = eval_env.step(act_result.action)
            total_reward += reward
            steps += 1
            done = payload["terminated"] or payload["truncated"]

        eval_env.close()

        env_info = payload["observation"]["env_info"]
        score = int(env_info["clean_score"])
        charges = int(env_info["charge_count"])

        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        episode_scores.append(score)
        episode_charges.append(charges)

        if recorder.frames:
            base = f"ep{ep + 1:02d}_score_{score}_step_{steps}"
            recorder.export_gif(eval_dir / f"{base}.gif", fps=10)
            recorder.export_log(eval_dir / f"{base}.log")

    _write_eval_summary(eval_dir, episode_rewards, episode_steps, episode_scores, episode_charges)

    return _build_result(episode_rewards, episode_steps)


def evaluate_multi_map_with_recording(
    algorithm: Algorithm,
    preprocessor: Preprocessor,
    map_configs: list[dict[str, Any]],
    map_names: list[str],
    eval_dir: Path,
    num_episodes: int = 10,
    gif_fps: int = 10,
) -> dict:
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[MapEvalResult] = []

    for map_config, map_name in zip(map_configs, map_names):
        map_dir = eval_dir / map_name
        map_dir.mkdir(parents=True, exist_ok=True)

        result = MapEvalResult(map_name=map_name)

        for ep in range(num_episodes):
            preprocessor.reset()
            recorder = TrajectoryRecorder()
            eval_env = create_env(
                map_config,
                enable_recording=True,
                trajectory_recorder=recorder,
                render_mode=None,
            )
            payload = eval_env.reset(options={"mode": "eval"})
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                map_img, vector, legal_action, reward = preprocessor.feature_process(
                    payload, preprocessor.curr_action
                )
                act_result = algorithm.act(map_img, vector, np.asarray(legal_action, dtype=np.float32), mode="exploit")
                payload = eval_env.step(act_result.action)
                total_reward += reward
                steps += 1
                done = payload["terminated"] or payload["truncated"]

            eval_env.close()

            env_info = payload["observation"]["env_info"]
            score = int(env_info["clean_score"])
            charges = int(env_info["charge_count"])

            result.rewards.append(total_reward)
            result.steps.append(steps)
            result.scores.append(score)
            result.charges.append(charges)

            if recorder.frames:
                base = f"ep{ep + 1:02d}_score_{score}_step_{steps}"
                recorder.export_gif(map_dir / f"{base}.gif", fps=gif_fps)
                recorder.export_log(map_dir / f"{base}.log")

        _write_map_summary(map_dir, map_name, result)
        all_results.append(result)

    _write_overall_summary(eval_dir, all_results)

    all_rewards = [r for res in all_results for r in res.rewards]
    all_scores = [s for res in all_results for s in res.scores]
    all_steps = [s for res in all_results for s in res.steps]
    all_charges = [c for res in all_results for c in res.charges]

    return {
        "results": all_results,
        "overall_avg_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "overall_avg_score": float(np.mean(all_scores)) if all_scores else 0.0,
        "overall_avg_steps": float(np.mean(all_steps)) if all_steps else 0.0,
        "overall_avg_charges": float(np.mean(all_charges)) if all_charges else 0.0,
        "total_episodes": len(all_rewards),
    }


def _build_result(
    episode_rewards: list[float],
    episode_steps: list[int],
) -> dict:
    return {
        "avg_reward": float(np.mean(episode_rewards)),
        "avg_steps": float(np.mean(episode_steps)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "episodes": len(episode_rewards),
    }


def _write_eval_summary(
    eval_dir: Path,
    rewards: list[float],
    steps: list[int],
    scores: list[int],
    charges: list[int],
) -> None:
    r = np.array(rewards, dtype=np.float32)
    s = np.array(steps, dtype=np.float32)
    sc = np.array(scores, dtype=np.float32)
    ch = np.array(charges, dtype=np.float32)

    lines = [
        "=== Evaluation Summary ===",
        f"episodes={len(rewards)}",
        "",
        f"Reward:  avg={r.mean():.2f} std={r.std():.2f} min={r.min():.2f} max={r.max():.2f}",
        f"Steps:   avg={s.mean():.1f} std={s.std():.1f} min={s.min():.0f} max={s.max():.0f}",
        f"Score:   avg={sc.mean():.1f} std={sc.std():.1f} min={sc.min():.0f} max={sc.max():.0f}",
        f"Charges: avg={ch.mean():.2f} std={ch.std():.2f} min={ch.min():.0f} max={ch.max():.0f}",
        "",
        "Per-episode:",
        "ep\treward\tsteps\tscore\tcharges",
    ]
    for i, (rw, st, sc_val, ch_val) in enumerate(zip(rewards, steps, scores, charges)):
        lines.append(f"{i + 1}\t{rw:.2f}\t{st:.0f}\t{sc_val:.0f}\t{ch_val:.0f}")

    (eval_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_map_summary(
    map_dir: Path,
    map_name: str,
    result: MapEvalResult,
) -> None:
    r = np.array(result.rewards, dtype=np.float32)
    s = np.array(result.steps, dtype=np.float32)
    sc = np.array(result.scores, dtype=np.float32)
    ch = np.array(result.charges, dtype=np.float32)

    lines = [
        f"=== {map_name} Evaluation Summary ===",
        f"episodes={result.num_episodes}",
        "",
        f"Reward:  avg={r.mean():.2f} std={r.std():.2f} min={r.min():.2f} max={r.max():.2f}",
        f"Steps:   avg={s.mean():.1f} std={s.std():.1f} min={s.min():.0f} max={s.max():.0f}",
        f"Score:   avg={sc.mean():.1f} std={sc.std():.1f} min={sc.min():.0f} max={sc.max():.0f}",
        f"Charges: avg={ch.mean():.2f} std={ch.std():.2f} min={ch.min():.0f} max={ch.max():.0f}",
        "",
        "Per-episode:",
        "ep\treward\tsteps\tscore\tcharges",
    ]
    for i, (rw, st, sc_val, ch_val) in enumerate(zip(result.rewards, result.steps, result.scores, result.charges)):
        lines.append(f"{i + 1}\t{rw:.2f}\t{st:.0f}\t{sc_val:.0f}\t{ch_val:.0f}")

    (map_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_overall_summary(
    eval_dir: Path,
    results: list[MapEvalResult],
) -> None:
    all_rewards = [r for res in results for r in res.rewards]
    all_scores = [s for res in results for s in res.scores]
    all_steps = [s for res in results for s in res.steps]
    all_charges = [c for res in results for c in res.charges]

    r = np.array(all_rewards, dtype=np.float32) if all_rewards else np.array([], dtype=np.float32)
    sc = np.array(all_scores, dtype=np.float32) if all_scores else np.array([], dtype=np.float32)
    s = np.array(all_steps, dtype=np.float32) if all_steps else np.array([], dtype=np.float32)
    ch = np.array(all_charges, dtype=np.float32) if all_charges else np.array([], dtype=np.float32)

    lines = [
        "=== Overall Multi-Map Evaluation Summary ===",
        f"maps={len(results)}  total_episodes={len(all_rewards)}",
        "",
        "--- Per-Map Averages ---",
        f"{'map':<12} {'episodes':>8} {'avg_reward':>11} {'avg_score':>10} {'avg_steps':>10} {'avg_charges':>12}",
    ]
    for res in results:
        lines.append(
            f"{res.map_name:<12} {res.num_episodes:>8} "
            f"{res.avg_reward:>11.2f} {res.avg_score:>10.1f} "
            f"{res.avg_steps:>10.1f} {res.avg_charges:>12.2f}"
        )

    lines.append("")
    lines.append("--- Overall Averages ---")
    if len(r) > 0:
        lines.append(f"Reward:  avg={r.mean():.2f} std={r.std():.2f} min={r.min():.2f} max={r.max():.2f}")
        lines.append(f"Score:   avg={sc.mean():.1f} std={sc.std():.1f} min={sc.min():.0f} max={sc.max():.0f}")
        lines.append(f"Steps:   avg={s.mean():.1f} std={s.std():.1f} min={s.min():.0f} max={s.max():.0f}")
        lines.append(f"Charges: avg={ch.mean():.2f} std={ch.std():.2f} min={ch.min():.0f} max={ch.max():.0f}")

    (eval_dir / "summary_all.txt").write_text("\n".join(lines), encoding="utf-8")
