"""
drone_env/models.py
# Typed Pydantic models for the Drone Delivery Env.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class DroneAction(BaseModel):
    direction: str = Field(
        default="WAIT",
        description="Movement direction: UP | DOWN | LEFT | RIGHT | WAIT",
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Task to load on reset: drone_env.graders.easy:grade_easy | medium | hard",
    )


class DroneObservation(BaseModel):
    grid: List[str] = Field(description="Emoji grid rows")
    grid_width: int
    grid_height: int
    drone_x: int
    drone_y: int
    battery: float = Field(description="Battery level 0.0–1.0")
    battery_steps_remaining: int
    deliveries_total: int
    deliveries_done: int
    current_target: Optional[Tuple[int, int]] = None
    distance_to_target: Optional[float] = None
    step_count: int
    max_steps: int
    reward_last: float
    reward_total: float
    score: float = 0.0
    done: bool
    message: str = ""
    legend: Dict[str, str] = Field(default_factory=dict)
    cell_types: List[List[str]] = Field(default_factory=list)


class DroneState(BaseModel):
    episode_id: str = ""
    task_name: str = "drone_env.graders.easy:grade_easy"
    step_count: int = 0
    done: bool = False
    reward_total: float = 0.0
    deliveries_done: int = 0
    deliveries_total: int = 0
    drone_x: int = 0
    drone_y: int = 0
    battery: float = 1.0
    path_history: List[Dict[str, Any]] = Field(default_factory=list)
