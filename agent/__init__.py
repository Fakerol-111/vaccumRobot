"""Reinforcement learning framework with multi-algorithm support.

Package structure:
    base.py              - Algorithm ABC, ActResult, LossInfo
    registry.py          - Algorithm registry (register / get / list_available)
    common/              - Shared components (Checkpoint, network building blocks)
    nn/                  - Network architectures (ActorCritic)
    ppo/                 - PPO algorithm implementation
    preprocessor.py      - Environment-specific feature engineering & reward

Usage:
    from agent import Algorithm, PPOAlgorithm, get_algorithm
    algo_cls = get_algorithm("ppo")
    algorithm = algo_cls(config, device)
"""

from agent.base import ActResult, Algorithm, LossInfo
from agent.grpo import GRPOAlgorithm
from agent.ppo import PPOAlgorithm
from agent.registry import get as get_algorithm, list_available, register

__all__ = [
    "Algorithm",
    "ActResult",
    "LossInfo",
    "GRPOAlgorithm",
    "PPOAlgorithm",
    "get_algorithm",
    "list_available",
    "register",
]
