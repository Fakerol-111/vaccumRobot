from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from agent.agent import Agent
from agent.definition import RolloutBatch
from metrics import MetricsLogger


def prepare_batches(segment: RolloutBatch, mini_batch_size: int) -> list[RolloutBatch]:
    n = len(segment)
    indices = np.random.permutation(n)
    batches: list[RolloutBatch] = []
    for start in range(0, n, mini_batch_size):
        end = start + mini_batch_size
        idx = indices[start:end]
        batches.append(segment[idx])
    return batches


class Trainer:
    def __init__(
        self,
        ppo_config,
        env_kwargs: dict,
        artifacts_dir: Path,
        map_name: str,
        device: torch.device | None = None,
    ):
        self.config = ppo_config
        self.env_kwargs = env_kwargs
        self.artifacts_dir = artifacts_dir
        self.map_name = map_name
        self.device = device or torch.device("cpu")
        self.map_dir = self.artifacts_dir / map_name
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.agent = Agent(ppo_config, self.device)

    # ---------- main entry ----------

    def train(self) -> None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = self.map_dir / "checkpoints" / run_id
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(log_file=self.checkpoint_dir / "train.log")

        self.logger._emit(f"Run: {run_id}")
        self.logger._emit(f"Map: {self.map_name}  timesteps={self.config.total_timesteps}")
        self.logger._emit(f"batch_size={self.config.batch_size}  lr={self.config.learning_rate}")
        self.logger._emit("")

        env = self._create_env()
        self._reset_collectors()
        self._episode_steps = 0
        self._episode_reward = 0.0
        global_step, start_time = 0, time.time()
        payload = env.reset(options={"mode": "train"})
        self.agent.preprocessor.reset()

        while global_step < self.config.total_timesteps:
            payload, global_step = self._collect_batch(env, payload, global_step)

            if self._buffer_is_full():
                rollout = self._build_rollout_batch(payload)
                loss_info = self._train_step(rollout)
                self._clear_collectors()
                self._record_update(rollout, loss_info)
                self._log_progress(global_step, start_time)

            if self._should_checkpoint(global_step):
                self._save_checkpoint(global_step)

        env.close()
        self._save_checkpoint(global_step)
        self.logger._emit(f"Model saved to {self.checkpoint_dir}")
        self._final_summary(start_time, global_step)
        self.logger.close()

    # ---------- private helpers ----------

    def _create_env(self):
        from env import GridWorldEnv
        return GridWorldEnv(**self.env_kwargs, enable_recording=False, render_mode=None)

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
        self, env, payload: dict, global_step: int,
    ) -> tuple[dict, int]:
        remaining = self.config.batch_size

        while remaining > 0 and global_step < self.config.total_timesteps:
            map_img, vector, legal_mask, reward = self.agent.preprocessor.feature_process(
                payload, self.agent.preprocessor.curr_action
            )
            action, log_prob, value = self.agent.forward_features(map_img, vector, legal_mask)
            payload = env.step(action)
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
                self.logger.log_episode(cleaned, self._episode_steps, charges, self._episode_reward)
                self._episode_steps = 0
                self._episode_reward = 0.0
                payload = env.reset(options={"mode": "train"})
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
        self.agent.save(path)

    def _final_summary(self, start_time: float, global_step: int) -> None:
        total_time = time.time() - start_time
        self.logger._emit(f"Total time: {total_time:.1f}s  Total steps: {global_step}")
        self.logger.print_training_summary()
