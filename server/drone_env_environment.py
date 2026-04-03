import random
from uuid import uuid4
from typing import Optional, List, Tuple

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DroneAction, DroneObservation
except ImportError:
    from models import DroneAction, DroneObservation

# Import map generator
try:
    from .map_generator import generate_grid
except ImportError:
    from map_generator import generate_grid

class DroneEnvironment(Environment):
    """
    Drone Environment Implementation.
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    global_latest_obs: Optional[DroneObservation] = None

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.difficulty = "easy"
        
        self.grid: List[List[str]] = []
        self.drone_pos: Tuple[int, int] = (0, 0)
        self.targets: List[Tuple[int, int]] = []
        self.obstacles: List[Tuple[int, int]] = []
        
        self.battery = 1.0
        self.distance = 0
        self.targets_delivered = 0
        self.total_targets = 0
        self.reward_total = 0.0
        self.is_done = False
        self.message = ""
        self.last_obs: Optional[DroneObservation] = None

    def reset(self) -> DroneObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Difficulties
        difficulties = ["easy", "medium", "hard"]
        self.difficulty = difficulties[self._reset_count % 3]

        grid, drone_pos, targets, obstacles = generate_grid(self.difficulty)
        
        self.grid = grid
        self.drone_pos = drone_pos
        self.targets = targets
        self.obstacles = obstacles
        self.total_targets = len(targets)
        self.targets_delivered = 0
        
        self.battery = 1.0
        self.distance = 0
        self.reward_total = 0.0
        self.is_done = False
        self.message = "Environment ready."
        
        obs = DroneObservation(
            grid=["".join(row) for row in self.grid],
            grid_width=len(self.grid[0]),
            grid_height=len(self.grid),
            step_count=0,
            reward_total=0.0,
            reward_last=0.0,
            deliveries_done=0,
            deliveries_total=self.total_targets,
            battery=1.0,
            message=self.message,
            done=False
        )
        self.last_obs = obs
        DroneEnvironment.global_latest_obs = obs
        return obs

    def step(self, action: DroneAction) -> DroneObservation:
        if self.is_done:
            return self.last_obs

        self._state.step_count += 1
        x, y = self.drone_pos
        dx, dy = 0, 0
        
        move = action.direction.upper()
        if move == "UP": dy = -1
        elif move == "DOWN": dy = 1
        elif move == "LEFT": dx = -1
        elif move == "RIGHT": dx = 1
        
        nx, ny = x + dx, y + dy
        height = len(self.grid)
        width = len(self.grid[0]) if height > 0 else 0
        
        reward = 0.0
        self.message = f"Done {move.lower()}"
        
        # Move costs battery
        self.battery -= 0.025
        self.distance += 1

        # Check bounds
        if 0 <= nx < width and 0 <= ny < height:
            cell = self.grid[ny][nx]
            
            if cell == "🏢" or cell == "🌳":
                self.message = "Crashed into building! 🚫"
                reward -= 0.1
            elif cell == "🚧":
                self.message = "Hit obstacle! 🚧"
                reward -= 0.2
                self.battery -= 0.1
            else:
                self.grid[self.drone_pos[1]][self.drone_pos[0]] = "🛣️"
                self.drone_pos = (nx, ny)
                if (nx, ny) in self.targets:
                    self.targets.remove((nx, ny))
                    self.targets_delivered += 1
                    reward += (1.0 / self.total_targets)
                    self.message = "Target Delivered! ✅"
                self.grid[ny][nx] = "🚁"
        else:
            self.message = "Hit the boundary!"
            reward -= 0.05
            
        if self.battery <= 0:
            self.battery = 0
            self.is_done = True
            self.message = "Battery empty! 🔋"
            
        if self.targets_delivered == self.total_targets:
            self.is_done = True
            self.message = "Mission Success! 🎉"
            
        self.reward_total += reward
        obs = DroneObservation(
            grid=["".join(row) for row in self.grid],
            grid_width=width,
            grid_height=height,
            step_count=self._state.step_count,
            reward_total=self.reward_total,
            reward_last=reward,
            deliveries_done=self.targets_delivered,
            deliveries_total=self.total_targets,
            battery=max(0, self.battery),
            message=self.message,
            done=self.is_done
        )
        self.last_obs = obs
        DroneEnvironment.global_latest_obs = obs
        return obs


    @property
    def state(self) -> State:
        return self._state

