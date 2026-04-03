"""
drone_delivery_env/core/state_manager.py
Handles initialization and management of the DroneState.
"""
import uuid
from typing import List, Tuple
from models import DroneState


def new_episode_state(
    task: str,
    deliveries: List[Tuple[int, int]],
    start_x: int,
    start_y: int,
    battery_max: float
) -> DroneState:
    """
    Creates a fresh DroneState for a new episode.
    """
    return DroneState(
        episode_id=str(uuid.uuid4()),
        task_name=task,
        step_count=0,
        done=False,
        reward_total=0.0,
        deliveries_done=0,
        deliveries_total=len(deliveries),
        drone_x=start_x,
        drone_y=start_y,
        battery=1.0,
        path_history=[]
    )
