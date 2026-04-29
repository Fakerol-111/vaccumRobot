from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Optimizer

from agent.definition import RolloutBatch
from agent.model import ActorCritic


class PPO:
    def __init__(
        self,
        model: ActorCritic,
        optimizer: Optimizer,
        config,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        map_imgs = torch.as_tensor(batch.map_imgs, dtype=torch.float32, device=self.device)
        vectors = torch.as_tensor(batch.vectors, dtype=torch.float32, device=self.device)
        legal_masks = torch.as_tensor(batch.legal_masks, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(batch.log_probs, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        old_values = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, new_values = self.model(map_imgs, vectors, legal_masks)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        ratio = torch.exp(new_log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = old_values + torch.clamp(
            new_values.squeeze() - old_values, -self.config.clip_epsilon, self.config.clip_epsilon
        )
        value_loss_unclipped = (new_values.squeeze() - returns) ** 2
        value_loss_clipped = (value_pred_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        entropy_loss = -entropy.mean()

        loss = policy_loss + self.config.value_coef * value_loss + self.config.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": loss.item(),
        }
