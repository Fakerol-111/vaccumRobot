"""PPO training framework with Actor-Learner separation.

Modules:
    model.py        - CNN+MLP Actor-Critic network, outputs action_logits & value
    preprocessor.py - feature engineering and reward computation
    definition.py   - data protocol, GAE computation, value bootstrapping
    algorithm.py    - PPO update (takes model + optimizer)
    agent.py        - unified predict / exploit / learn / save / load
    checkpoint.py   - checkpoint format (model, optimizer, step, config, RNG)
"""
