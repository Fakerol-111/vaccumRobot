"""On-policy experience buffer for PPO."""

from __future__ import annotations

import numpy as np

from agent.ppo.batch import RolloutBatch


class RolloutBuffer:
    """Stores a segment of on-policy experience for PPO updates."""

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.map_imgs: list[np.ndarray] = []
        self.vectors: list[np.ndarray] = []
        self.legal_masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def append(
        self,
        map_img: np.ndarray,
        vector: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ) -> None:
        self.map_imgs.append(map_img)
        self.vectors.append(vector)
        self.legal_masks.append(np.array(legal_mask, dtype=np.float32))
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self) -> int:
        return len(self.rewards)

    def to_batch(
        self,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> RolloutBatch:
        return RolloutBatch(
            map_imgs=np.stack(self.map_imgs),
            vectors=np.stack(self.vectors),
            legal_masks=np.stack(self.legal_masks),
            actions=np.array(self.actions, dtype=np.int64),
            log_probs=np.array(self.log_probs, dtype=np.float32),
            values=np.array(self.values, dtype=np.float32),
            rewards=np.array(self.rewards, dtype=np.float32),
            dones=np.array(self.dones, dtype=np.int8),
            advantages=advantages,
            returns=returns,
        )
