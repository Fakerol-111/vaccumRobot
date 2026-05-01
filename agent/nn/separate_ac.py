"""Separate Actor and Critic networks (no shared encoder).

Architecture mirrors ``ActorCritic`` but each network has its own
independent map encoder, vector encoder, and fusion layer.

Usage:
    model = SeparateActorCritic(num_actions=8)
    logits, value = model(map_img, vector, legal_mask)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from agent.common.functional import make_conv, make_fc


class ActorNetwork(nn.Module):
    """Standalone policy network with its own encoder."""

    def __init__(self, num_actions: int = 8, actor_gain: float = 0.01):
        super().__init__()

        self.map_encoder = nn.Sequential(
            make_conv(9, 32),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            make_conv(32, 64, stride=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            make_conv(64, 96, stride=2),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
            make_conv(96, 96),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
        )
        self.map_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.map_proj = nn.Sequential(
            nn.Flatten(),
            make_fc(96 * 3 * 3, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.vector_encoder = nn.Sequential(
            make_fc(10, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            make_fc(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            make_fc(256 + 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.policy_trunk = nn.Sequential(
            make_fc(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            make_fc(128, 64),
            nn.SiLU(inplace=True),
        )

        self.actor_head = make_fc(64, num_actions, gain=actor_gain)

    def forward(
        self,
        map_img: torch.Tensor,
        vector: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        map_feat = self.map_encoder(map_img)
        map_feat = self.map_pool(map_feat)
        map_feat = self.map_proj(map_feat)

        vec_feat = self.vector_encoder(vector)
        fused = self.fusion(torch.cat([map_feat, vec_feat], dim=-1))

        hidden = self.policy_trunk(fused)
        logits = self.actor_head(hidden)

        if legal_mask is not None:
            logits = logits.masked_fill(legal_mask == 0, -1e9)

        return logits


class CriticNetwork(nn.Module):
    """Standalone value network with its own encoder."""

    def __init__(self):
        super().__init__()

        self.map_encoder = nn.Sequential(
            make_conv(9, 32),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            make_conv(32, 64, stride=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            make_conv(64, 96, stride=2),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
            make_conv(96, 96),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
        )
        self.map_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.map_proj = nn.Sequential(
            nn.Flatten(),
            make_fc(96 * 3 * 3, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.vector_encoder = nn.Sequential(
            make_fc(10, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            make_fc(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            make_fc(256 + 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.value_trunk = nn.Sequential(
            make_fc(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            make_fc(128, 64),
            nn.SiLU(inplace=True),
        )

        self.critic_head = make_fc(64, 1, gain=1.0)

    def forward(
        self,
        map_img: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        map_feat = self.map_encoder(map_img)
        map_feat = self.map_pool(map_feat)
        map_feat = self.map_proj(map_feat)

        vec_feat = self.vector_encoder(vector)
        fused = self.fusion(torch.cat([map_feat, vec_feat], dim=-1))

        hidden = self.value_trunk(fused)
        value = self.critic_head(hidden)
        return value


class SeparateActorCritic(nn.Module):
    """Combined actor-critic with completely independent networks.

    Interface-compatible with ``ActorCritic`` — ``forward`` returns
    ``(logits, value)``.
    """

    def __init__(self, num_actions: int = 8, actor_gain: float = 0.01):
        super().__init__()
        self.actor = ActorNetwork(num_actions, actor_gain=actor_gain)
        self.critic = CriticNetwork()

    def forward(
        self,
        map_img: torch.Tensor,
        vector: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ):
        logits = self.actor(map_img, vector, legal_mask)
        value = self.critic(map_img, vector)
        return logits, value
