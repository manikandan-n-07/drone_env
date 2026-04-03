"""
drone_env/core/drone.py
# Physics and movement kinematics for the Drone Delivery Env.
"""
from typing import Tuple


def compute_next_pos(x: int, y: int, direction: str) -> Tuple[int, int]:
    """
    Calculate the next grid coordinates for the drone.
    direction: UP | DOWN | LEFT | RIGHT | WAIT
    """
    move = direction.upper()
    if move == "UP":    return (x, y - 1)
    if move == "DOWN":  return (x, y + 1)
    if move == "LEFT":  return (x - 1, y)
    if move == "RIGHT": return (x + 1, y)
    return (x, y)


def drain_battery(current: float, cost: float) -> float:
    """
    Apply battery drain and clamp at 0.0.
    """
    return max(0.0, current - cost)
