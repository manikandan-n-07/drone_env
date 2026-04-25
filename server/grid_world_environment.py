"""
server/grid_world_environment.py
Core Drone Delivery RL Environment — refactored to use core/ modules.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        def reset(self): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        @property
        def state(self): raise NotImplementedError

try:
    from drone_env.models import DroneAction, DroneObservation, DroneState
    from drone_env.core.grid_generator import generate_city_map, EMOJI, LEGEND
    from drone_env.core.tasks import TASK_CONFIG
    from drone_env.core.drone import compute_next_pos, drain_battery
    from drone_env.core.obstacles import check_move
    from drone_env.core.state_manager import new_episode_state
except ImportError:
    from models import DroneAction, DroneObservation, DroneState
    from core.grid_generator import generate_city_map, EMOJI, LEGEND
    from core.tasks import TASK_CONFIG
    from core.drone import compute_next_pos, drain_battery
    from core.obstacles import check_move
    from core.state_manager import new_episode_state
try:
    from drone_env.graders import GRADERS
except ImportError:
    from graders import GRADERS
try:
    from drone_env.rl.trainer import record_episode
except ImportError:
    from rl.trainer import record_episode


class DroneDeliveryEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._cfg = TASK_CONFIG["graders:grade_easy"]
        self._grid: List[List[str]] = []
        self._deliveries: List[Tuple[int, int]] = [] # (x, y)
        self._delivery_status: List[str] = [] # "pending", "assigned", "picked", "done"
        self._rng = torch.Generator()
        self._step_records: List[Dict] = []
        self._start_pos: Tuple[int, int] = (0, 0)
        self._state = DroneState()
        self._current_session = None # Session lock
        # Initialize with default task
        self.reset()

    def reset(self, action: Optional[DroneAction] = None) -> DroneObservation:
        if self._step_records and self._state.episode_id and not self._state.done:
             self._persist_episode()

        _TASK_ID_MAP = {
            "easy_delivery":   "graders:grade_easy",
            "medium_delivery": "graders:grade_medium",
            "hard_delivery":   "graders:grade_hard",
        }
        task = "graders:grade_easy"
        if action and action.task_name:
            name = action.task_name
            if name in _TASK_ID_MAP:
                task = _TASK_ID_MAP[name]
            elif name in TASK_CONFIG:
                task = name

        self._cfg = TASK_CONFIG[task]
        self._rng.manual_seed(random.randint(0, 2**31))

        self._grid, self._deliveries, self._start_pos = generate_city_map(self._cfg, self._rng)
        self._delivery_status = ["pending"] * len(self._deliveries)
        self._step_records = []

        self._state = new_episode_state(
            task=task,
            deliveries=self._deliveries,
            start_pos=self._start_pos,
            battery_max=self._cfg["battery_max"],
            n_drones=self._cfg["n_drones"]
        )
        if action and action.session_id:
            self._current_session = action.session_id
            print(f"[Env] Session locked to: {self._current_session}")

        print(f"[Env] Reset complete. Task: {task}, Drones: {len(self._state.drones)}")
        return self._build_obs(0.01, f"🚀 Mission started! {len(self._state.drones)} drones ready at Godown.")

    def step(self, action: DroneAction) -> DroneObservation:
        if self._state.done:
            return self._build_obs(0.01, "Episode ended. Call reset().")

        cfg = self._cfg
        
        # Check session lock
        if self._current_session and action.session_id and action.session_id != self._current_session:
            print(f"[Env] Ignoring step from unauthorized session: {action.session_id}")
            return self._build_obs(0.0, "Unauthorized session")
            
        print(f"[Env] Raw action.actions: {action.actions}, Legacy direction: {action.direction}")
        
        # Merge legacy direction into actions for Drone 0 if provided
        raw_actions = dict(action.actions) if action.actions else {}
        if action.direction and 0 not in raw_actions:
            raw_actions[0] = action.direction
            
        # Ensure keys are integers
        actions = {int(k): v for k, v in raw_actions.items()}
        print(f"[Env] Unified actions: {actions}")
        
        total_reward = 0.0
        messages = []

        # 1. Assignment Phase (Heuristic: nearest idle drone to pending package)
        for d in self._state.drones:
            if not d.has_package and d.target_id is None:
                # Find nearest pending delivery
                best_id = -1
                min_dist = float('inf')
                for i, (tx, ty) in enumerate(self._deliveries):
                    if self._delivery_status[i] == "pending":
                        dist = abs(d.x - tx) + abs(d.y - ty)
                        if dist < min_dist:
                            min_dist = dist
                            best_id = i
                
                if best_id != -1:
                    d.target_id = best_id
                    self._delivery_status[best_id] = "assigned"
                    messages.append(f"🛰️ Fleet AI: Drone {d.id} tracking package {best_id}")

        # 2. Movement & Interaction Phase
        
        for d in self._state.drones:
            move = actions.get(d.id, "WAIT").upper()
            print(f"[Env] Drone {d.id} at ({d.x}, {d.y}) moving {move} (Target: {d.target_id})")
            nx, ny = compute_next_pos(d.x, d.y, move)
            outcome, cell = check_move(self._grid, nx, ny, cfg["width"], cfg["height"])

            drone_reward = cfg["r_step"] # Base small step penalty

            if outcome == "ok" or (outcome in ["building", "tree"] and cell != "wall"):
                d.x, d.y = nx, ny
                if outcome in ["building", "tree"]:
                    drone_reward = cfg.get(f"r_{outcome}", 0.05)
            else:
                drone_reward = cfg.get(f"r_{outcome}", 0.05)

            # Interaction
            if not d.has_package and d.target_id is not None:
                tx, ty = self._deliveries[d.target_id]
                if d.x == tx and d.y == ty:
                    d.has_package = True
                    self._delivery_status[d.target_id] = "picked"
                    drone_reward = cfg["r_pickup"]
                    messages.append(f"🚁 Drone {d.id}: Package 📦 at ({tx}, {ty}) is now DONE ✅! Reporting back to Godown 🏭.")
            elif d.has_package and d.target_id is not None:
                # Return to Godown
                if d.x == self._start_pos[0] and d.y == self._start_pos[1]:
                    self._delivery_status[d.target_id] = "done"
                    self._state.deliveries_done += 1
                    d.has_package = False
                    d.target_id = None
                    drone_reward = cfg["r_delivery"]
                    messages.append(f"✅ Drone {d.id}: Docked at Godown 🏭. Delivery cycle finalized.")

            # Battery
            d.battery = drain_battery(d.battery * cfg["battery_max"], cfg["battery_cost"]) / cfg["battery_max"]
            if d.battery <= 0:
                drone_reward = cfg["r_battery_dead"]
            
            total_reward += drone_reward

        self._state.step_count += 1
        num_drones = len(self._state.drones)
        if num_drones > 0:
            avg_reward = max(0.01, min(0.99, total_reward / num_drones))
        else:
            avg_reward = 0.01
        self._state.reward_total += avg_reward

        # Check Done
        done = False
        if self._state.deliveries_done == len(self._deliveries):
            done = True
            msg = "🎉 Mission Success! All packages at Godown."
        elif any(d.battery <= 0 for d in self._state.drones):
            done = True
            msg = "🔋 Critical Failure: A drone ran out of battery!"
        elif self._state.step_count >= cfg["max_steps"]:
            done = True
            msg = "⏰ Timeout reached."
        else:
            msg = " | ".join(messages) if messages else "📡 Network Synchronized. Fleet in transit..."

        self._state.done = done
        if done:
            self._persist_episode()

        return self._build_obs(avg_reward, msg)

    @property
    def state(self) -> DroneState:
        return self._state

    def _persist_episode(self):
        try:
            record_episode(
                task=self._state.task_name,
                steps=[], # Path history simplified for multi-agent
                grid_meta=dict(width=self._cfg["width"], height=self._cfg["height"]),
                delivery_positions=[list(d) for d in self._deliveries],
                deliveries_done=self._state.deliveries_done,
                total_reward=float(round(self._state.reward_total, 4)),
            )
        except Exception: pass

    def _build_obs(self, reward: float, message: str) -> DroneObservation:
        cfg = self._cfg
        print(f"[Env] Building obs for drones: {self._state.drones}")
        return DroneObservation(
            grid=self._render_grid(),
            grid_width=int(cfg["width"]), grid_height=int(cfg["height"]),
            drones=self._state.drones,
            targets=self._deliveries,
            deliveries_total=len(self._deliveries),
            deliveries_done=self._state.deliveries_done,
            step_count=self._state.step_count,
            max_steps=cfg["max_steps"],
            reward_last=float(reward),
            reward_total=float(self._state.reward_total),
            score=float(max(0.01, min(0.99, round(GRADERS[self._state.task_name](self._state), 4)))),
            done=self._state.done,
            message=message,
            legend=dict(LEGEND),
            cell_types=list(self._grid),
            x=self._state.drones[0].x if self._state.drones else 0,
            y=self._state.drones[0].y if self._state.drones else 0,
            drone_x=self._state.drones[0].x if self._state.drones else 0,
            drone_y=self._state.drones[0].y if self._state.drones else 0,
            battery=self._state.drones[0].battery if self._state.drones else 1.0,
            current_target=self._deliveries[self._state.drones[0].target_id] if (self._state.drones and self._state.drones[0].target_id is not None) else (0, 0),
            distance_to_target=self._compute_dist_to_target(0)
        )

    def _compute_dist_to_target(self, drone_idx: int) -> float:
        if not self._state.drones or drone_idx >= len(self._state.drones):
            return 0.0
        d = self._state.drones[drone_idx]
        if d.target_id is None:
            return 0.0
        tx, ty = self._deliveries[d.target_id]
        return float(abs(d.x - tx) + abs(d.y - ty))

    @property
    def graders(self) -> Dict:
        """Expose graders for the environment."""
        return GRADERS

    def _render_grid(self) -> List[str]:
        rows = []
        for y, row in enumerate(self._grid):
            line = ""
            for x, cell in enumerate(row):
                # Check for drones first
                drone_here = None
                for d in self._state.drones:
                    if d.x == x and d.y == y:
                        drone_here = d
                        break
                
                if drone_here:
                    line += EMOJI["drone"]
                    continue
                
                # Check for delivery targets
                is_delivery = False
                for i, (tx, ty) in enumerate(self._deliveries):
                    if x == tx and y == ty:
                        status = self._delivery_status[i]
                        if status in ["done", "picked"]:
                            line += EMOJI["done_del"]
                        elif status in ["pending", "assigned"]:
                            line += EMOJI["delivery"]
                        is_delivery = True
                        break
                
                if not is_delivery:
                    line += EMOJI.get(cell, cell)
            rows.append(line)
        return rows
