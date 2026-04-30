from __future__ import annotations

import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from agent.base import Algorithm
from agent.common.checkpoint import build_config_snapshot, restore_rng_state
from agent.preprocessor import Preprocessor
from configs.map_loader import load_map_config
from core.paths import get_checkpoints_root, get_checkpoint_path, get_run_info_path, get_run_dir, get_train_log_path
from env.factory import create_env
from services.metrics_service import MetricsLogger

if TYPE_CHECKING:
    from services.dashboard_service import MetricsCollector


def _get_git_info() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
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
        algorithm: Algorithm,
        preprocessor: Preprocessor,
        algo_config,
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
        metrics_config: dict[str, Any] | None = None,
    ):
        self.algorithm = algorithm
        self.preprocessor = preprocessor
        self.config = algo_config
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
        self._metrics_config = metrics_config or {"max_updates": 500, "max_episodes": 500}

        self._env: Any = None
        self._current_env_config: dict[str, Any] | None = None
        self._current_map_idx = 0
        self._current_map_id = 0
        self._map_name = ""
        self._current_stage_name = ""
        self._env_config_cache: dict[tuple, dict] = {}
        self._last_save_time = 0.0
        self._last_done = True

    # ---------- public properties ----------

    @property
    def run_id(self) -> str:
        assert self._run_id is not None, (
            "run_id not set. Use core.trainer_runner.run_training() or "
            "pass run_id= explicitly."
        )
        return self._run_id

    # ---------- main entry ----------

    def train(self) -> None:
        if self._resume_from is not None:
            self._resume_training(self._resume_from)
            return

        self.checkpoint_dir = get_run_dir(get_checkpoints_root(self.artifacts_dir), self.run_id)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(
            log_file=get_train_log_path(self.checkpoint_dir),
            collector=self._collector,
            max_updates=self._metrics_config["max_updates"],
            max_episodes=self._metrics_config["max_episodes"],
        )
        try:
            self._write_source_config()
            self._write_run_meta(self.run_id)

            self._print_config_summary(self.run_id)

            self._env = self._create_initial_env()
            self._episode_steps = 0
            self._episode_reward = 0.0
            global_step, start_time = 0, time.time()
            payload = self._env.reset(seed=self._base_seed + self._episode_counter, options={"mode": "train"})
            self.preprocessor.reset()

            try:
                while global_step < self.config.total_timesteps:
                    self._log_stage_transition(global_step)
                    payload, global_step, algo_updated = self._collect_batch(payload, global_step)

                    bootstrap_state = None
                    if not self._last_done:
                        m, v, l, _ = self.preprocessor.feature_process(payload, self.preprocessor.curr_action)
                        bootstrap_state = (m, v, l)
                    if algo_updated or self.algorithm.maybe_update(bootstrap_state) is not None:
                        self._log_progress(global_step, start_time)

                    if self._should_checkpoint(global_step):
                        self._save_checkpoint(global_step)
            except KeyboardInterrupt:
                self.logger._emit("")
                self.logger._emit(">>> KeyboardInterrupt received, saving checkpoint ...")
                self._save_checkpoint(global_step)
                self._final_summary(start_time, global_step)
                self.logger._emit(">>> Training stopped by user.")
                return

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
        self._current_env_config = dict(config)
        self._current_map_id = map_id
        self._map_name = f"map_{map_id}"
        self.algorithm.set_env_config(self._current_env_config)
        return create_env(config, enable_recording=False, render_mode=None)

    def _create_next_env(self, global_step: int):
        config, map_id = self._pick_next_map(global_step)
        self._current_env_config = dict(config)
        self._current_map_id = map_id
        self._map_name = f"map_{map_id}"
        self.algorithm.set_env_config(self._current_env_config)
        return create_env(config, enable_recording=False, render_mode=None)

    def _write_run_meta(self, run_id: str) -> None:
        c = self.config
        meta = {
            "run_id": run_id,
            "seed": self._base_seed,
            "algo": {
                "learning_rate": c.learning_rate,
                "gamma": c.gamma,
                "max_grad_norm": c.max_grad_norm,
                "total_timesteps": c.total_timesteps,
                "save_interval": c.save_interval,
                "log_interval": c.log_interval,
                "num_actions": c.num_actions,
                "local_view_size": c.local_view_size,
                "max_npcs": c.max_npcs,
                "gae_lambda": getattr(c, "gae_lambda", None),
                "clip_epsilon": getattr(c, "clip_epsilon", None),
                "value_coef": getattr(c, "value_coef", None),
                "entropy_coef": getattr(c, "entropy_coef", None),
                "ppo_epochs": getattr(c, "ppo_epochs", None),
                "batch_size": getattr(c, "batch_size", None),
                "mini_batch_size": getattr(c, "mini_batch_size", None),
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
        meta_path = get_run_info_path(self.checkpoint_dir)
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
        self.logger._emit(f"  Algorithm: {type(self.algorithm).__name__.replace('Algorithm', '').upper()}")
        self.logger._emit("=" * 65)
        self.logger._emit("")
        self.logger._emit("--- Environment ---")
        self.logger._emit(f"  Maps:          {self._default_map_list}")
        self.logger._emit(f"  Map strategy:  {self.map_strategy}")
        self.logger._emit(f"  NPC count:     {self.default_npc_count}")
        self.logger._emit(f"  Station count: {self.default_station_count}")
        self.logger._emit("")
        self.logger._emit("--- Algorithm Hyperparameters ---")
        self.logger._emit(f"  Learning rate:  {c.learning_rate}")
        self.logger._emit(f"  Gamma:          {c.gamma}")
        if hasattr(c, "gae_lambda"):
            self.logger._emit(f"  GAE lambda:     {c.gae_lambda}")
        if hasattr(c, "clip_epsilon"):
            self.logger._emit(f"  Clip epsilon:   {c.clip_epsilon}")
        if hasattr(c, "value_coef"):
            self.logger._emit(f"  Value coef:     {c.value_coef}")
        if hasattr(c, "entropy_coef"):
            self.logger._emit(f"  Entropy coef:   {c.entropy_coef}")
        self.logger._emit(f"  Max grad norm:  {c.max_grad_norm}")
        self.logger._emit("")
        self.logger._emit("--- Training ---")
        self.logger._emit(f"  Seed:            {self._base_seed}")
        self.logger._emit(f"  Total timesteps: {c.total_timesteps}")
        if hasattr(c, "batch_size"):
            self.logger._emit(f"  Batch size:      {c.batch_size}")
        if hasattr(c, "mini_batch_size"):
            self.logger._emit(f"  Mini batch:      {c.mini_batch_size}")
        if hasattr(c, "ppo_epochs"):
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
        if self.curriculum_enabled:
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
            algo_name = type(self.algorithm).__name__.replace("Algorithm", "").lower()
            self._collector.set_run_info({
                "algo": algo_name,
                "run_id": run_id,
                "seed": self._base_seed,
                "total_timesteps": c.total_timesteps,
                "map_strategy": self.map_strategy,
                "curriculum_enabled": self.curriculum_enabled,
                "maps": self._default_map_list,
                "batch_size": getattr(c, "batch_size", None),
                "lr": c.learning_rate,
            })
            self._collector.add_event("info", {"algo": algo_name, "run_id": run_id, "seed": self._base_seed, "message": f"Run started: {run_id}  seed={self._base_seed}"})
            info_lines = [
                f"Maps: {self._default_map_list}  strategy={self.map_strategy}",
                f"NPC={self.default_npc_count}  Stations={self.default_station_count}",
                f"gamma={c.gamma}" + (f"  lambda={c.gae_lambda}" if hasattr(c, "gae_lambda") else "")
                              + (f"  clip={c.clip_epsilon}" if hasattr(c, "clip_epsilon") else ""),
            ]
            if hasattr(c, "value_coef"):
                info_lines.append(f"value_coef={c.value_coef}  entropy_coef={c.entropy_coef}  max_grad_norm={c.max_grad_norm}")
            else:
                info_lines.append(f"max_grad_norm={c.max_grad_norm}")
            if hasattr(c, "gae_lambda"):
                info_lines.append(f"batch={c.batch_size}  mini_batch={c.mini_batch_size}  epochs={c.ppo_epochs}")
            info_lines.extend([
                f"total_steps={c.total_timesteps}  save_interval={c.save_interval}  log_interval={c.log_interval}",
                f"lr={c.learning_rate}  device={self.device}",
                f"Curriculum: {'enabled' if self.curriculum_enabled else 'disabled'}",
            ])
            for s in info_lines:
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

    def _should_checkpoint(self, global_step: int) -> bool:
        if global_step <= 0:
            return False
        time_interval = self.config.save_time_interval
        if time_interval > 0:
            return time.time() - self._last_save_time >= time_interval
        return global_step % self.config.save_interval == 0

    def _collect_batch(
        self, payload: dict, global_step: int,
    ) -> tuple[dict, int, bool]:
        remaining = self.config.batch_size
        algo_updated = False

        while remaining > 0 and global_step < self.config.total_timesteps:
            map_img, vector, legal_mask, reward = self.preprocessor.feature_process(
                payload, self.preprocessor.curr_action
            )
            legal_mask = np.asarray(legal_mask, dtype=np.float32)

            # 为分支算法（如 GRPO）保存 env + preprocessor 快照
            if hasattr(self.algorithm, "set_branch_state"):
                self.algorithm.set_branch_state({
                    "env": self._env.get_state(),
                    "pp": self.preprocessor.get_state(),
                })

            result = self.algorithm.act(map_img, vector, legal_mask, mode="explore")
            payload = self._env.step(result.action)
            done = bool(payload["terminated"] or payload["truncated"])

            ret = self.algorithm.on_step(
                map_img, vector, np.array(legal_mask, dtype=np.float32),
                result.action, result.log_prob, result.value,
                reward, done,
            )
            if ret is not None:
                algo_updated = True

            self._last_done = done

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
                self.preprocessor.reset()

        return payload, global_step, algo_updated

    def _log_progress(self, global_step: int, start_time: float) -> None:
        if global_step % self.config.log_interval != 0:
            return
        elapsed = time.time() - start_time
        fps = global_step / max(elapsed, 1)
        algo_line = ""
        reporter = self.algorithm.metrics_reporter
        if reporter is not None:
            algo_line = "  " + reporter.update_summary()
        self.logger.print_summary(global_step, fps, algo_summary=algo_line)

    def _save_checkpoint(self, global_step: int) -> None:
        path = get_checkpoint_path(self.checkpoint_dir, global_step)
        extra = {
            "default_npc_count": self.default_npc_count,
            "default_station_count": self.default_station_count,
            "map_strategy": self.map_strategy,
            "curriculum_enabled": self.curriculum_enabled,
            "curriculum_stages": self.curriculum_stages,
            "default_map_list": self._default_map_list,
            "seed": self._base_seed,
        }
        config_snapshot = build_config_snapshot(self.config, extra)
        self.algorithm.save_checkpoint(
            path=path,
            global_step=global_step,
            episode_counter=self._episode_counter,
            current_map_idx=self._current_map_idx,
            current_map_id=self._current_map_id,
            current_stage_name=self._current_stage_name,
            config_snapshot=config_snapshot,
        )
        self._last_save_time = time.time()

    def _final_summary(self, start_time: float, global_step: int) -> None:
        total_time = time.time() - start_time
        self.logger._emit(f"Total time: {total_time:.1f}s  Total steps: {global_step}")
        self.logger.print_training_summary()
        reporter = self.algorithm.metrics_reporter
        if reporter is not None:
            for line in reporter.final_summary_lines():
                self.logger._emit(f"  {line}")

    def _resume_training(self, checkpoint_path: Path) -> None:
        ckpt = self.algorithm.load_checkpoint(checkpoint_path)
        global_step = ckpt.global_step
        self._episode_counter = ckpt.episode_counter
        self._current_map_idx = ckpt.current_map_idx
        self._current_map_id = ckpt.current_map_id
        self._current_stage_name = ckpt.current_stage_name

        restore_rng_state(ckpt.rng_state)

        self.checkpoint_dir = get_run_dir(get_checkpoints_root(self.artifacts_dir), self.run_id)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(
            log_file=get_train_log_path(self.checkpoint_dir),
            collector=self._collector,
            max_updates=self._metrics_config["max_updates"],
            max_episodes=self._metrics_config["max_episodes"],
        )
        try:
            self.logger._emit("=" * 65)
            self.logger._emit(f"  Resuming run [{self.run_id}] from checkpoint step {global_step}")
            self.logger._emit(f"  Checkpoint: {checkpoint_path}")
            self.logger._emit("=" * 65)

            self._env = self._create_next_env(global_step)
            self._episode_steps = 0
            self._episode_reward = 0.0
            start_time = time.time()
            payload = self._env.reset(seed=self._base_seed + self._episode_counter, options={"mode": "train"})
            self.preprocessor.reset()

            try:
                while global_step < self.config.total_timesteps:
                    self._log_stage_transition(global_step)
                    payload, global_step, algo_updated = self._collect_batch(payload, global_step)

                    bootstrap_state = None
                    if not self._last_done:
                        m, v, l, _ = self.preprocessor.feature_process(payload, self.preprocessor.curr_action)
                        bootstrap_state = (m, v, l)
                    if algo_updated or self.algorithm.maybe_update(bootstrap_state) is not None:
                        self._log_progress(global_step, start_time)

                    if self._should_checkpoint(global_step):
                        self._save_checkpoint(global_step)
            except KeyboardInterrupt:
                self.logger._emit("")
                self.logger._emit(">>> KeyboardInterrupt received, saving checkpoint ...")
                self._save_checkpoint(global_step)
                self._final_summary(start_time, global_step)
                self.logger._emit(">>> Training stopped by user.")
                return

            self._save_checkpoint(global_step)
            self.logger._emit(f"Model saved to {self.checkpoint_dir}")
            self._final_summary(start_time, global_step)
        finally:
            if self._env is not None:
                self._env.close()
            self.logger.close()
