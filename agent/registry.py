"""Algorithm registry: map algorithm names to Algorithm subclasses.

Usage:
    from agent.registry import register, get, list_available

    @register("ppo")
    class PPOAlgorithm(Algorithm):
        ...

    # Later, in trainer_runner:
    algo_cls = get(config.algo_name)
    algorithm = algo_cls(algo_config, device)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.base import Algorithm

_registry: dict[str, type[Algorithm]] = {}


def register(name: str):
    """Decorator that registers an Algorithm subclass under *name*."""
    def decorator(cls: type[Algorithm]) -> type[Algorithm]:
        _registry[name] = cls
        return cls
    return decorator


def get(name: str) -> type[Algorithm]:
    """Return the Algorithm class registered under *name*."""
    if name not in _registry:
        available = ", ".join(sorted(_registry))
        raise ValueError(
            f"Unknown algorithm {name!r}. Available: [{available}]"
        )
    return _registry[name]


def list_available() -> list[str]:
    """Return a sorted list of registered algorithm names."""
    return sorted(_registry)
