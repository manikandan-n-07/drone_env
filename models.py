"""
drone_env/models.py
# Typed Pydantic models for the Drone Delivery Env.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class DroneAction(BaseModel):
    actions: Dict[int, str] = Field(
        default_factory=dict,
        description="Map of drone ID to movement direction: UP | DOWN | LEFT | RIGHT | WAIT",
    )
    direction: Optional[str] = Field(
        default=None,
        description="Legacy field for single-agent movement (maps to drone 0)"
    )
    task_name: Optional[str] = Field(
        default=None,
        description="Task to load on reset: graders:grade_easy | medium | hard",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Active session ID to prevent interference"
    )


class DroneInfo(BaseModel):
    id: int
    x: int
    y: int
    battery: float
    has_package: bool = False
    target_id: Optional[int] = None


class DroneObservation(BaseModel):
    grid: List[str] = Field(default_factory=list, description="Emoji grid rows")
    grid_width: int = 0
    grid_height: int = 0
    drones: List[DroneInfo] = Field(default_factory=list)
    targets: List[Tuple[int, int]] = Field(default_factory=list, description="Coordinates of all delivery targets")
    deliveries_total: int = 0
    deliveries_done: int = 0
    step_count: int = 0
    max_steps: int = 0
    reward_last: float = 0.0
    reward_total: float = 0.0
    score: float = 0.0
    done: bool = False
    message: str = ""
    legend: Dict[str, str] = Field(default_factory=dict)
    cell_types: List[List[str]] = Field(default_factory=list)
    # Backward compatibility
    x: Optional[int] = 0
    y: Optional[int] = 0
    drone_x: Optional[int] = 0
    drone_y: Optional[int] = 0
    battery: Optional[float] = 1.0
    current_target: Optional[Tuple[int, int]] = (0, 0)
    distance_to_target: Optional[float] = 0.0

    # Backward compatibility for old UI
    @property
    def x(self) -> int: return self.drones[0].x if self.drones else 0
    @property
    def y(self) -> int: return self.drones[0].y if self.drones else 0
    @property
    def battery(self) -> float: return self.drones[0].battery if self.drones else 1.0


class DroneState(BaseModel):
    episode_id: str = ""
    task_name: str = "graders:grade_easy"
    step_count: int = 0
    done: bool = False
    reward_total: float = 0.0
    deliveries_done: int = 0
    deliveries_total: int = 0
    drones: List[DroneInfo] = Field(default_factory=list)
    path_history: List[Dict[str, Any]] = Field(default_factory=list)
