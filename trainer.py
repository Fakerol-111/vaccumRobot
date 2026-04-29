from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from agent.agent import Agent
from agent.checkpoint import build_config_snapshot, restore_rng_state
from agent.definition import RolloutBatch
from config.map_loader import load_map_config
from metrics import MetricsLogger

if TYPE_CHECKING:
    from training_dashboard import MetricsCollector


def prepare_batches(segment: RolloutBatch, mini_batch_size: int) -> list[RolloutBatch]:
    n = len(segment)
    indices = np.random.permutation(n)
    batches: list[RolloutBatch] = []
    for start in range(0, n, mini_batch_size):
        end = start + mini_batch_size
        idx = indices[start:end]
        batches.append(segment[idx])
    return batches


def _find_nearest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    ckpt_files = list(checkpoint_dir.glob("checkpoint_*.pt"))
    if not ckpt_files:
        return None
    def _step_from_name(p: Path) -> int:
        m = re.search(r"checkpoint_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else 0
    ckpt_files.sort(key=_step_from_name, reverse=True)
    return ckpt_files[0]


def _get_git_info() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    git_dir = repo_root / ".git"
    info: dict[str, Any] = {"commit": None, "dirty": None, "source": None}

    if git_dir.is_dir():
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["commit"] = result.stdout.strip()
        except Exception:
            pass

        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                info["dirty"] = bool(result.stdout.strip())
        except Exception:
            pass

    if info["commit"]:
        info["source"] = "git"
    else:
        info["commit"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        info["dirty"] = True
        info["source"] = "timestamp"

    return info


class Trainer:
    def __init__(
        self,
        ppo_config,
        default_npc_count: int,
        default_station_count: int,
        map_strategy: str,
        curriculum: dict[str, Any],
        artifacts_dir: Path,
        device: torch.device | None = None,
        collector: MetricsCollector | None = None,
        default_map_list: list[int] | None = None,
        seed: int = 42,
        resume_from: str | Path | None = None,
        run_id: str | None = None,
        config_path: Path | None = None,
    ):
        self.config = ppo_config
        self.default_npc_count = default_npc_count
        self.default_station_count = default_station_count
        self.map_strategy = map_strategy
        self.curriculum_enabled = curriculum["enabled"]
        self.curriculum_stages: list[dict[str, Any]] = curriculum.get("stages") or []
        self.artifacts_dir = artifacts_dir
        self.device = device or torch.device("cpu")
        self._collector = collector
        self._default_map_list = default_map_list or [1, 2, 3, 4]
        self._base_seed = seed
        self._episode_counter = 0
        self._resume_from = Path(resume_from) if resume_from else None
        self._run_id = run_id
        self._config_path = config_path

        self.map_dir = self.artifacts_dir / "multi_map"
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.agent = Agent(ppo_config, self.device)

        self._env: Any = None
        self._current_map_idx = 0
        self._current_map_id = 0
        self._map_name = ""
        self._current_stage_name = ""
        self._env_config_cache: dict[tuple, dict] = {}

    # ---------- main entry ----------

    def train(self) -> None:
        if self._resume_from is not None:
            self._resume_training(self._resume_from)
            return

        run_id = self._run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.map_dir / "checkpoints" / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(log_file=self.checkpoint_dir / "train.log", collector=self._collector)
        try:
            self._write_source_config()
            self._write_run_meta(run_id)

            self._print_config_summary(run_id)

            self._env = self._create_initial_env()
            self._reset_collectors()
            self._episode_steps = 0
            self._episode_reward = 0.0
            global_step, start_time = 0, time.time()
            payload = self._env.reset(seed=self._base_seed + self._episode_counter, options={"mode": "train"})
            self.agent.preprocessor.reset()

            while global_step < self.config.total_timesteps:
                self._log_stage_transition(global_step)
                payload, global_step = self._collect_batch(payload, global_step)

                if self._buffer_is_full():
                    rollout = self._build_rollout_batch(payload)
                    loss_info = self._train_step(rollout)
                    self._clear_collectors()
                    self._record_update(rollout, loss_info)
                    self._log_progress(global_step, start_time)

                if self._should_checkpoint(global_step):
                    self._save_checkpoint(global_step)

            self._save_checkpoint(global_step)
            self.logger._emit(f"Model saved to {self.checkpoint_dir}")
            self._final_summary(start_time, global_step)
        finally:
            if self._env is not None:
                self._env.close()
            self.logger.close()

    # ---------- curriculum / map helpers ----------

    def _resolve_curriculum_stage(self, global_step: int) -> dict[str, Any] | None:
        if not self.curriculum_enabled or not self.curriculum_stages:
            return None
        for stage in self.curriculum_stages:
            if global_step < stage["total_steps"]:
                return stage
        return self.curriculum_stages[-1]

    def _get_env_config(self, map_id: int, npc_count: int, station_count: int) -> dict[str, Any]:
        cache_key = (map_id, npc_count, station_count)
        if cache_key not in self._env_config_cache:
            cfg = load_map_config(map_id)
            cfg["npc_count"] = npc_count
            cfg["station_count"] = station_count
            self._env_config_cache[cache_key] = cfg
        return self._env_config_cache[cache_key]

    def _pick_next_map(self, global_step: int) -> tuple[dict[str, Any], int]:
        stage = self._resolve_curriculum_stage(global_step)
        if stage:
            map_ids = stage["maps"]
            npc_count = stage["npc_count"]
            station_count = stage["station_count"]
        else:
            map_ids = self._default_map_list
            npc_count = self.default_npc_count
            station_count = self.default_station_count

        if self.map_strategy == "random":
            map_id = int(np.random.choice(map_ids))
        else:
            n = len(map_ids)
            map_id = map_ids[self._current_map_idx % n]
            self._current_map_idx = (self._current_map_idx + 1) % n

        env_config = self._get_env_config(map_id, npc_count, station_count)
        return env_config, map_id

    def _create_initial_env(self):
        config, map_id = self._pick_next_map(0)
        self._current_map_id = map_id
        self._map_name = f"map_{map_id}"
        return GridWorldEnv(**config, enable_recording=False, render_mode=None)

    def _create_next_env(self, global_step: int):
        config, map_id = self._pick_next_map(global_step)
        self._current_map_id = map_id
        self._map_name = f"map_{map_id}"
        return GridWorldEnv(**config, enable_recording=False, render_mode=None)

    def _write_run_meta(self, run_id: str) -> None:
        c = self.config
        meta = {
            "run_id": run_id,
            "seed": self._base_seed,
            "ppo": {
                "learning_rate": c.learning_rate,
                "gamma": c.gamma,
                "gae_lambda": c.gae_lambda,
                "clip_epsilon": c.clip_epsilon,
                "value_coef": c.value_coef,
                "entropy_coef": c.entropy_coef,
                "max_grad_norm": c.max_grad_norm,
                "ppo_epochs": c.ppo_epochs,
                "batch_size": c.batch_size,
                "mini_batch_size": c.mini_batch_size,
                "total_timesteps": c.total_timesteps,
                "save_interval": c.save_interval,
                "log_interval": c.log_interval,
                "num_actions": c.num_actions,
                "local_view_size": c.local_view_size,
                "max_npcs": c.max_npcs,
            },
            "env": {
                "maps": self._default_map_list,
                "map_strategy": self.map_strategy,
                "npc_count": self.default_npc_count,
                "station_count": self.default_station_count,
            },
            "curriculum": {
                "enabled": self.curriculum_enabled,
                "stages": self.curriculum_stages,
            },
            "device": str(self.device),
            "git": _get_git_info(),
        }
        meta_path = self.checkpoint_dir / "run_info.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _write_source_config(self) -> None:
        src = self._config_path
        if src is None or not src.exists():
            self.logger._emit(f"[config] No source config to copy (path={src})")
            return
        dst = self.checkpoint_dir / src.name
        shutil.copy2(str(src), str(dst))
        self.logger._emit(f"[config] Copied source config to {dst}")

    def _print_config_summary(self, run_id: str) -> None:
        c = self.config
        self.logger._emit("=" * 65)
        self.logger._emit(f"  Run: {run_id}")
        self.logger._emit("=" * 65)
        self.logger._emit("")
        self.logger._emit("--- Environment ---")
        self.logger._emit(f"  Maps:          {self._default_map_list}")
        self.logger._emit(f"  Map strategy:  {self.map_strategy}")
        self.logger._emit(f"  NPC count:     {self.default_npc_count}")
        self.logger._emit(f"  Station count: {self.default_station_count}")
        self.logger._emit("")
        self.logger._emit("--- PPO Hyperparameters ---")
        self.logger._emit(f"  Learning rate:  {c.learning_rate}")
        self.logger._emit(f"  Gamma:          {c.gamma}")
        self.logger._emit(f"  GAE lambda:     {c.gae_lambda}")
        self.logger._emit(f"  Clip epsilon:   {c.clip_epsilon}")
        self.logger._emit(f"  Value coef:     {c.value_coef}")
        self.logger._emit(f"  Entropy coef:   {c.entropy_coef}")
        self.logger._emit(f"  Max grad norm:  {c.max_grad_norm}")
        self.logger._emit("")
        self.logger._emit("--- Training ---")
        self.logger._emit(f"  Seed:            {self._base_seed}")
        self.logger._emit(f"  Total timesteps: {c.total_timesteps}")
        self.logger._emit(f"  Batch size:      {c.batch_size}")
        self.logger._emit(f"  Mini batch:      {c.mini_batch_size}")
        self.logger._emit(f"  PPO epochs:      {c.ppo_epochs}")
        self.logger._emit(f"  Save interval:   {c.save_interval}")
        self.logger._emit(f"  Log interval:    {c.log_interval}")
        self.logger._emit(f"  Device:          {self.device}")
        self.logger._emit("")
        self.logger._emit("--- Model ---")
        self.logger._emit(f"  Local view: {c.local_view_size}x{c.local_view_size}")
        self.logger._emit(f"  Actions:    {c.num_actions}")
        self.logger._emit(f"  Max NPCs:   {c.max_npcs}")
        self.logger._emit("")
        self.logger._emit("--- Curriculum ---")
        self.logger._emit(f"  Enabled: {self.curriculum_enabled}")
        if self.curriculum_stages:
            for i, s in enumerate(self.curriculum_stages):
                self.logger._emit(
                    f"  Stage {i+1} [{s['name']}]: maps={s['maps']} "
                    f"npc={s['npc_count']} stations={s['station_count']} "
                    f"until_step={s['total_steps']}"
                )
        self.logger._emit("")
        self.logger._emit("=" * 65)
        self.logger._emit("")

        if self._collector is not None:
            self._collector.set_run_info({
                "run_id": run_id,
                "seed": self._base_seed,
                "total_timesteps": c.total_timesteps,
                "map_strategy": self.map_strategy,
                "curriculum_enabled": self.curriculum_enabled,
                "maps": self._default_map_list,
                "batch_size": c.batch_size,
                "lr": c.learning_rate,
            })
            self._collector.add_event("info", {"run_id": run_id, "seed": self._base_seed, "message": f"Run started: {run_id}  seed={self._base_seed}"})
            for s in [
                f"Maps: {self._default_map_list}  strategy={self.map_strategy}",
                f"NPC={self.default_npc_count}  Stations={self.default_station_count}",
                f"gamma={c.gamma}  lambda={c.gae_lambda}  clip={c.clip_epsilon}",
                f"value_coef={c.value_coef}  entropy_coef={c.entropy_coef}  max_grad_norm={c.max_grad_norm}",
                f"batch={c.batch_size}  mini_batch={c.mini_batch_size}  epochs={c.ppo_epochs}",
                f"total_steps={c.total_timesteps}  save_interval={c.save_interval}  log_interval={c.log_interval}",
                f"lr={c.learning_rate}  device={self.device}",
                f"Curriculum: {'enabled' if self.curriculum_enabled else 'disabled'}",
            ]:
                self._collector.add_event("info", {"message": s})

    def _log_stage_transition(self, global_step: int) -> None:
        stage = self._resolve_curriculum_stage(global_step)
        stage_name = stage["name"] if stage else "default"
        if stage_name != self._current_stage_name:
            self._current_stage_name = stage_name
            self._current_map_idx = 0
            self.logger._emit(f"[Curriculum] >>> Entering stage: {stage_name} at step {global_step}")
            if self._collector is not None:
                self._collector.add_event("stage", {
                    "stage_name": stage_name,
                    "step": global_step,
                })

    # ---------- private helpers ----------

    def _reset_collectors(self) -> None:
        self._map_imgs: list[np.ndarray] = []
        self._vectors: list[np.ndarray] = []
        self._legal_masks: list[np.ndarray] = []
        self._actions: list[int] = []
        self._log_probs_list: list[float] = []
        self._values: list[float] = []
        self._rewards: list[float] = []
        self._dones: list[int] = []

    def _clear_collectors(self) -> None:
        self._map_imgs.clear()
        self._vectors.clear()
        self._legal_masks.clear()
        self._actions.clear()
        self._log_probs_list.clear()
        self._values.clear()
        self._rewards.clear()
        self._dones.clear()

    def _buffer_is_full(self) -> bool:
        return len(self._rewards) == self.config.batch_size

    def _should_checkpoint(self, global_step: int) -> bool:
        return global_step % self.config.save_interval == 0 and global_step > 0

    def _collect_batch(
        self, payload: dict, global_step: int,
    ) -> tuple[dict, int]:
        remaining = self.config.batch_size

        while remaining > 0 and global_step < self.config.total_timesteps:
            map_img, vector, legal_mask, reward = self.agent.preprocessor.feature_process(
                payload, self.agent.preprocessor.curr_action
            )
            action, log_prob, value = self.agent.forward_features(map_img, vector, legal_mask)
            payload = self._env.step(action)
            done = int(payload["terminated"] or payload["truncated"])

            self._map_imgs.append(map_img)
            self._vectors.append(vector)
            self._legal_masks.append(np.array(legal_mask, dtype=np.float32))
            self._actions.append(action)
            self._log_probs_list.append(log_prob)
            self._values.append(value)
            self._rewards.append(reward)
            self._dones.append(done)

            self._episode_reward += reward
            self._episode_steps += 1
            remaining -= 1
            global_step += 1

            if done:
                env_info = payload["observation"]["env_info"]
                cleaned = int(env_info["clean_score"])
                charges = int(env_info["charge_count"])
                self.logger.log_episode(cleaned, self._episode_steps, charges, self._episode_reward, self._map_name)
                self._episode_steps = 0
                self._episode_reward = 0.0
                self._episode_counter += 1

                self._env.close()
                self._env = self._create_next_env(global_step)
                payload = self._env.reset(seed=self._base_seed + self._episode_counter, options={"mode": "train"})
                self.agent.preprocessor.reset()

        return payload, global_step

    def _build_rollout_batch(self, payload: dict) -> RolloutBatch:
        rewards_arr = np.array(self._rewards, dtype=np.float32)
        values_arr = np.array(self._values, dtype=np.float32)
        dones_arr = np.array(self._dones, dtype=np.int8)

        bootstrap_value = 0.0
        if not bool(dones_arr[-1]):
            bootstrap_value = self.agent.get_bootstrap_value(payload)

        advantages, returns_arr = self.agent.compute_gae(
            rewards_arr, values_arr, dones_arr, bootstrap_value,
        )

        return RolloutBatch(
            map_imgs=np.stack(self._map_imgs),
            vectors=np.stack(self._vectors),
            legal_masks=np.stack(self._legal_masks),
            actions=np.array(self._actions, dtype=np.int64),
            log_probs=np.array(self._log_probs_list, dtype=np.float32),
            values=np.array(self._values, dtype=np.float32),
            rewards=rewards_arr,
            dones=dones_arr,
            advantages=advantages,
            returns=returns_arr,
        )

    def _train_step(self, rollout: RolloutBatch) -> dict[str, float]:
        batches = prepare_batches(rollout, self.config.mini_batch_size)
        return self.agent.learn(batches)

    def _record_update(self, rollout: RolloutBatch, loss_info: dict[str, float]) -> None:
        mean_reward = float(np.mean(rollout.rewards))
        self.logger.log_update(
            mean_reward,
            loss_info["policy_loss"],
            loss_info["value_loss"],
            loss_info["entropy"],
        )

    def _log_progress(self, global_step: int, start_time: float) -> None:
        if global_step % self.config.log_interval != 0:
            return
        elapsed = time.time() - start_time
        fps = global_step / max(elapsed, 1)
        self.logger.print_summary(global_step, fps)

    def _save_checkpoint(self, global_step: int) -> None:
        path = self.checkpoint_dir / f"checkpoint_{global_step}.pt"
        config_snapshot = build_config_snapshot(self)
        self.agent.save_checkpoint(
            path=path,
            global_step=global_step,
            episode_counter=self._episode_counter,
            current_map_idx=self._current_map_idx,
            current_map_id=self._current_map_id,
            current_stage_name=self._current_stage_name,
            config_snapshot=config_snapshot,
        )

    def _final_summary(self, start_time: float, global_step: int) -> None:
        total_time = time.time() - start_time
        self.logger._emit(f"Total time: {total_time:.1f}s  Total steps: {global_step}")
        self.logger.print_training_summary()

    def _resume_training(self, checkpoint_path: Path) -> None:
        ckpt = self.agent.load_checkpoint(checkpoint_path)
        global_step = ckpt.global_step
        self._episode_counter = ckpt.episode_counter
        self._current_map_idx = ckpt.current_map_idx
        self._current_map_id = ckpt.current_map_id
        self._current_stage_name = ckpt.current_stage_name

        restore_rng_state(ckpt.rng_state)

        run_id = self._run_id or checkpoint_path.parent.name
        self.checkpoint_dir = self.map_dir / "checkpoints" / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(log_file=self.checkpoint_dir / "train.log", collector=self._collector)
        try:
            self.logger._emit("=" * 65)
            self.logger._emit(f"  Resuming run [{run_id}] from checkpoint step {global_step}")
            self.logger._emit(f"  Checkpoint: {checkpoint_path}")
            self.logger._emit("=" * 65)

            self._env = self._create_next_env(global_step)
            self._reset_collectors()
            self._episode_steps = 0
            self._episode_reward = 0.0
            start_time = time.time()
            payload = self._env.reset(seed=self._base_seed + self._episode_counter, options={"mode": "train"})
            self.agent.preprocessor.reset()

            while global_step < self.config.total_timesteps:
                self._log_stage_transition(global_step)
                payload, global_step = self._collect_batch(payload, global_step)

                if self._buffer_is_full():
                    rollout = self._build_rollout_batch(payload)
                    loss_info = self._train_step(rollout)
                    self._clear_collectors()
                    self._record_update(rollout, loss_info)
                    self._log_progress(global_step, start_time)

                if self._should_checkpoint(global_step):
                    self._save_checkpoint(global_step)

            self._save_checkpoint(global_step)
            self.logger._emit(f"Model saved to {self.checkpoint_dir}")
            self._final_summary(start_time, global_step)
        finally:
            if self._env is not None:
                self._env.close()
            self.logger.close()


from env import GridWorldEnv
