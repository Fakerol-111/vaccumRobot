"""Neural network architectures.

Modules:
    actor_critic.py    - Two-stream Actor-Critic network (shared encoder)
    separate_ac.py     - Separate Actor and Critic networks (independent encoders)
    factory.py         - ``create_model()`` — build model by type name
"""

from agent.nn.factory import create_model
from agent.nn.separate_ac import SeparateActorCritic

__all__ = ["create_model", "SeparateActorCritic"]
