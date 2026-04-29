from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from agent.agent import Agent


def evaluate(
    agent: Agent,
    env_kwargs: dict[str, Any],
    num_episodes: int = 3,
) -> dict:
    from env import GridWorldEnv

    eval_env = GridWorldEnv(**env_kwargs, enable_recording=False, render_mode=None)

    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    for _ in range(num_episodes):
        agent.preprocessor.reset()
        payload = eval_env.reset(options={"mode": "eval"})
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            map_img, vector, legal_action, reward = agent.preprocessor.feature_process(
                payload, agent.preprocessor.curr_action
            )
            action, _, _ = agent.forward_features(map_img, vector, legal_action, deterministic=True)
            payload = eval_env.step(action)
            total_reward += reward
            steps += 1
            done = payload["terminated"] or payload["truncated"]

        episode_rewards.append(total_reward)
        episode_steps.append(steps)

    eval_env.close()

    return _build_result(episode_rewards, episode_steps)


def evaluate_with_recording(
    agent: Agent,
    env_kwargs: dict[str, Any],
    eval_dir: Path,
    num_episodes: int = 10,
) -> dict:
    from env import GridWorldEnv, TrajectoryRecorder

    eval_dir.mkdir(parents=True, exist_ok=True)

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_scores: list[int] = []
    episode_charges: list[int] = []

    for ep in range(num_episodes):
        agent.preprocessor.reset()
        recorder = TrajectoryRecorder()
        eval_env = GridWorldEnv(
            **env_kwargs,
            enable_recording=True,
            trajectory_recorder=recorder,
            render_mode=None,
        )
        payload = eval_env.reset(options={"mode": "eval"})
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            map_img, vector, legal_action, reward = agent.preprocessor.feature_process(
                payload, agent.preprocessor.curr_action
            )
            action, _, _ = agent.forward_features(map_img, vector, legal_action, deterministic=True)
            payload = eval_env.step(action)
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
        f"=== Evaluation Summary ===",
        f"episodes={len(rewards)}",
        f"",
        f"Reward:  avg={r.mean():.2f} std={r.std():.2f} min={r.min():.2f} max={r.max():.2f}",
        f"Steps:   avg={s.mean():.1f} std={s.std():.1f} min={s.min():.0f} max={s.max():.0f}",
        f"Score:   avg={sc.mean():.1f} std={sc.std():.1f} min={sc.min():.0f} max={sc.max():.0f}",
        f"Charges: avg={ch.mean():.2f} std={ch.std():.2f} min={ch.min():.0f} max={ch.max():.0f}",
        f"",
        f"Per-episode:",
        f"ep\treward\tsteps\tscore\tcharges",
    ]
    for i, (rw, st, sc_val, ch_val) in enumerate(zip(rewards, steps, scores, charges)):
        lines.append(f"{i + 1}\t{rw:.2f}\t{st:.0f}\t{sc_val:.0f}\t{ch_val:.0f}")

    (eval_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
