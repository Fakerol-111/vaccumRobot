from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from agent.algorithm import PPO
from agent.checkpoint import Checkpoint, capture_rng_state
from agent.definition import RolloutBatch, compute_gae
from agent.model import ActorCritic
from agent.preprocessor import Preprocessor


class Agent:
    def __init__(self, config, device: torch.device | None = None):
        self.config = config
        self.device = device or torch.device("cpu")

        self.model = ActorCritic(num_actions=config.num_actions)
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.algorithm = PPO(self.model, self.optimizer, config, self.device)
        self.preprocessor = Preprocessor()

    def predict(
        self,
        payload: dict[str, Any],
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        map_img, vector, legal_action, _ = self.preprocessor.feature_process(
            payload, self.preprocessor.curr_action
        )
        return self.forward_features(map_img, vector, legal_action, deterministic)

    def forward_features(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_action: list[int],
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        map_img_t = torch.as_tensor(map_img, dtype=torch.float32, device=self.device).unsqueeze(0)
        vector_t = torch.as_tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        legal_t = torch.as_tensor(legal_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(map_img_t, vector_t, legal_t)
            dist = Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def get_bootstrap_value(
        self,
        payload: dict[str, Any],
    ) -> float:
        map_img, vector, legal_action, _ = self.preprocessor.feature_process(
            payload, self.preprocessor.curr_action
        )

        map_img_t = torch.as_tensor(map_img, dtype=torch.float32, device=self.device).unsqueeze(0)
        vector_t = torch.as_tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        legal_t = torch.as_tensor(legal_action, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            _, value = self.model(map_img_t, vector_t, legal_t)

        return value.item()

    def learn(self, batches: list[RolloutBatch]) -> dict[str, float]:
        self.model.train()
        epoch_losses: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [], "total_loss": []
        }

        for _ in range(self.config.ppo_epochs):
            for batch in batches:
                loss_info = self.algorithm.update(batch)
                for k, v in loss_info.items():
                    epoch_losses[k].append(v)

        self.model.eval()
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        bootstrap_value: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        return compute_gae(
            rewards, values, dones,
            self.config.gamma, self.config.gae_lambda,
            bootstrap_value,
        )

    def save(self, path: str | Path) -> None:
        self._save_file(Path(path))

    def load(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.to(self.device)

    def _save_file(self, path: Path) -> None:
        torch.save(self.model.state_dict(), str(path))

    def save_checkpoint(
        self,
        path: str | Path,
        global_step: int,
        episode_counter: int = 0,
        current_map_idx: int = 0,
        current_map_id: int = 0,
        current_stage_name: str = "",
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        ckpt = Checkpoint(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            global_step=global_step,
            episode_counter=episode_counter,
            current_map_idx=current_map_idx,
            current_map_id=current_map_id,
            current_stage_name=current_stage_name,
            config_snapshot=config_snapshot or {},
            rng_state=capture_rng_state(),
        )
        torch.save(ckpt.to_dict(), str(path))

    def load_checkpoint(self, path: str | Path) -> Checkpoint:
        data = torch.load(str(path), map_location=self.device, weights_only=True)
        ckpt = Checkpoint.from_dict(data)

        self.model.load_state_dict(ckpt.model_state_dict)
        self.model.to(self.device)
        self.optimizer.load_state_dict(ckpt.optimizer_state_dict)

        return ckpt
