from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RolloutBatch:
    map_imgs: np.ndarray
    vectors: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def __len__(self) -> int:
        return len(self.rewards)

    def __getitem__(self, idx) -> RolloutBatch:
        return RolloutBatch(
            map_imgs=self.map_imgs[idx],
            vectors=self.vectors[idx],
            legal_masks=self.legal_masks[idx],
            actions=np.atleast_1d(self.actions[idx]),
            log_probs=np.atleast_1d(self.log_probs[idx]),
            values=np.atleast_1d(self.values[idx]),
            rewards=np.atleast_1d(self.rewards[idx]),
            dones=np.atleast_1d(self.dones[idx]),
            advantages=np.atleast_1d(self.advantages[idx]),
            returns=np.atleast_1d(self.returns[idx]),
        )

    @classmethod
    def concatenate(cls, batches: list[RolloutBatch]) -> RolloutBatch:
        return cls(
            map_imgs=np.concatenate([b.map_imgs for b in batches], axis=0),
            vectors=np.concatenate([b.vectors for b in batches], axis=0),
            legal_masks=np.concatenate([b.legal_masks for b in batches], axis=0),
            actions=np.concatenate([b.actions for b in batches], axis=0),
            log_probs=np.concatenate([b.log_probs for b in batches], axis=0),
            values=np.concatenate([b.values for b in batches], axis=0),
            rewards=np.concatenate([b.rewards for b in batches], axis=0),
            dones=np.concatenate([b.dones for b in batches], axis=0),
            advantages=np.concatenate([b.advantages for b in batches], axis=0),
            returns=np.concatenate([b.returns for b in batches], axis=0),
        )


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
    bootstrap_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    last_value = bootstrap_value

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * last_value * (1.0 - dones[t]) - values[t]
        last_advantage = delta + gamma * gae_lambda * (1.0 - dones[t]) * last_advantage
        advantages[t] = last_advantage
        last_value = values[t]

    returns = advantages + values
    return advantages, returns
