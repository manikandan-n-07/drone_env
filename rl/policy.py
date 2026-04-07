"""
rl/policy.py
Epsilon-greedy policy for exploration during training.
"""
from __future__ import annotations
import random
import torch
from drone_env.rl.model import PathQNet, ACTIONS


class EpsilonGreedyPolicy:
    """
    Selects actions using epsilon-greedy strategy.
    Epsilon decays linearly from eps_start → eps_end over decay_steps.
    """

    def __init__(self, eps_start: float = 1.0, eps_end: float = 0.05, decay_steps: int = 5000):
        self.eps = eps_start
        self.eps_end = eps_end
        self.decay = (eps_start - eps_end) / max(decay_steps, 1)
        self.step_count = 0

    def select_action(self, q_values: torch.Tensor) -> int:
        if random.random() < self.eps:
            return random.randint(0, len(ACTIONS) - 1)
        return int(q_values.argmax().item())

    def decay_epsilon(self):
        self.step_count += 1
        self.eps = max(self.eps_end, self.eps - self.decay)

    @property
    def current_epsilon(self) -> float:
        return self.eps
