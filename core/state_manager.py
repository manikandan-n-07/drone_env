"""
drone_delivery_env/core/state_manager.py
Handles initialization and management of the DroneState.
"""
import uuid
from typing import List, Tuple
try:
    from drone_env.models import DroneState, DroneInfo
except ImportError:
    from models import DroneState, DroneInfo


def new_episode_state(
    task: str,
    deliveries: List[Tuple[int, int]],
    start_pos: Tuple[int, int],
    battery_max: float,
    n_drones: int = 1
) -> DroneState:
    """
    Creates a fresh DroneState for a new episode.
    """
    drones = []
    for i in range(n_drones):
        # Even further offset to ensure visibility on different cells
        offset_x = (i % 2) * 2
        offset_y = (i // 2) * 2
        drones.append(DroneInfo(
            id=i,
            x=start_pos[0] + offset_x,
            y=start_pos[1] + offset_y,
            battery=1.0,
            has_package=False,
            target_id=None
        ))

    return DroneState(
        episode_id=str(uuid.uuid4()),
        task_name=task,
        step_count=0,
        done=False,
        reward_total=0.0,
        deliveries_done=0,
        deliveries_total=len(deliveries),
        drones=drones,
        path_history=[]
    )
