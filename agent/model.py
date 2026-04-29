"""Two-stream Actor-Critic network for Robot Vacuum.

Input:
    map_img: (B, 9, 21, 21)
    vector_data: (B, 10)

Output:
    actor_logits: (B, NUM_ACTIONS)
    value: (B, 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_fc(in_dim, out_dim, gain=1.41421):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(layer.weight, gain=1.41421)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, num_actions: int = 8):
        super().__init__()

        self.map_encoder = nn.Sequential(
            _make_conv(9, 32),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            _make_conv(32, 64, stride=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            _make_conv(64, 96, stride=2),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
            _make_conv(96, 96),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
        )
        self.map_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.map_proj = nn.Sequential(
            nn.Flatten(),
            _make_fc(96 * 3 * 3, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.vector_encoder = nn.Sequential(
            _make_fc(10, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
            _make_fc(64, 64),
            nn.LayerNorm(64),
            nn.SiLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            _make_fc(256 + 64, 256),
            nn.LayerNorm(256),
            nn.SiLU(inplace=True),
        )

        self.policy_trunk = nn.Sequential(
            _make_fc(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            _make_fc(128, 64),
            nn.SiLU(inplace=True),
        )
        self.value_trunk = nn.Sequential(
            _make_fc(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(inplace=True),
            _make_fc(128, 64),
            nn.SiLU(inplace=True),
        )

        self.actor_head = _make_fc(64, num_actions, gain=0.01)
        self.critic_head = _make_fc(64, 1, gain=1.0)

    def forward(
        self,
        map_img: torch.Tensor,
        vector: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ):
        map_feat = self.map_encoder(map_img)
        map_feat = self.map_pool(map_feat)
        map_feat = self.map_proj(map_feat)

        vec_feat = self.vector_encoder(vector)
        fused = self.fusion(torch.cat([map_feat, vec_feat], dim=-1))

        policy_hidden = self.policy_trunk(fused)
        value_hidden = self.value_trunk(fused)

        action_logits = self.actor_head(policy_hidden)
        value = self.critic_head(value_hidden)

        if legal_mask is not None:
            action_logits = action_logits.masked_fill(legal_mask == 0, -1e9)

        return action_logits, value
