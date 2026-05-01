"""Model factory — select network architecture by name.

Each algorithm reads ``model_type`` from config and calls ``create_model()``
to instantiate the corresponding network.
"""

from __future__ import annotations

import torch.nn as nn

from agent.nn.actor_critic import ActorCritic
from agent.nn.separate_ac import SeparateActorCritic


def create_model(
    model_type: str,
    num_actions: int,
    actor_gain: float = 0.01,
) -> nn.Module:
    """Build an actor-critic network by type name.

    Returns a module whose ``forward(map_img, vector, legal_mask)``
    returns ``(logits, value)``.

    Supported types:
      - ``"shared"`` (default) — ``ActorCritic``, shared encoder
      - ``"separate"`` — ``SeparateActorCritic``, independent encoders
    """
    if model_type == "separate":
        return SeparateActorCritic(num_actions=num_actions, actor_gain=actor_gain)

    return ActorCritic(num_actions=num_actions, actor_gain=actor_gain)
