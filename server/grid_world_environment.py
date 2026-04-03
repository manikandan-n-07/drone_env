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

from drone_env.models import DroneAction, DroneObservation, DroneState
from drone_env.core.grid_generator import generate_city_map, EMOJI, LEGEND
from drone_env.core.tasks import TASK_CONFIG
from drone_env.core.drone import compute_next_pos, drain_battery
from drone_env.core.obstacles import check_move
from drone_env.core.state_manager import new_episode_state
from drone_env.core.graders import GRADERS
from drone_env.rl.trainer import record_episode


class DroneDeliveryEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._state = DroneState()
        self._cfg = TASK_CONFIG["easy_delivery"]
        self._grid: List[List[str]] = []
        self._deliveries: List[Tuple[int, int]] = []
        self._delivered: List[bool] = []
        self._drone_x = 0
        self._drone_y = 0
        self._battery = 0
        self._rng = torch.Generator()
        self._step_records: List[Dict] = []
        self._start_pos: Tuple[int, int] = (0, 0)

    def reset(self, action: Optional[DroneAction] = None) -> DroneObservation:
        # Potentially persist old episode if not already done
        if self._step_records and self._state.episode_id and not self._state.done:
             self._persist_episode()

        task = "easy_delivery"
        if action and action.task_name and action.task_name in TASK_CONFIG:
            task = action.task_name
        elif action and action.task_name:
            # handle cases like "easy" instead of "easy_delivery"
            for k in TASK_CONFIG:
                if k.startswith(action.task_name):
                    task = k; break

        self._cfg = TASK_CONFIG[task]
        self._rng.manual_seed(random.randint(0, 2**31))

        self._grid, self._deliveries, self._start_pos = generate_city_map(self._cfg, self._rng)
        self._delivered = [False] * len(self._deliveries)
        self._drone_x, self._drone_y = self._start_pos
        self._battery = self._cfg["battery_max"]
        self._step_records = []

        self._state = new_episode_state(
            task=task,
            deliveries=self._deliveries,
            start_x=self._drone_x,
            start_y=self._drone_y,
            battery_max=self._cfg["battery_max"],
        )
        return self._build_obs(0.0, "🚁 Mission started! Deliver all packages.")

    def step(self, action: DroneAction) -> DroneObservation:
        if self._state.done:
            return self._build_obs(0.0, "Episode ended. Call reset().")

        cfg = self._cfg
        direction = (action.direction or "WAIT").upper()
        nx, ny = compute_next_pos(self._drone_x, self._drone_y, direction)
        outcome, cell = check_move(self._grid, nx, ny, cfg["width"], cfg["height"])

        reward = cfg["r_step"]
        msg = ""

        # Reward shaping (Distance bonus)
        _, dist_before = self._next_target()
        
        if outcome == "wall":
            nx, ny = self._drone_x, self._drone_y
            reward += cfg["r_wall"]
            msg = f"Hit {direction} boundary! 🚫"
        elif outcome == "blocked":
            nx, ny = self._drone_x, self._drone_y
            reward += cfg["r_blocked"]
            msg = f"Path is {direction}ly blocked!"
        elif outcome == "building":
            self._drone_x, self._drone_y = nx, ny
            reward = cfg.get("r_building", -0.1)
            msg = f"Flying above a building! {EMOJI['building']}"
        elif outcome == "tree":
            self._drone_x, self._drone_y = nx, ny
            reward = cfg.get("r_tree", -0.1)
            msg = f"Flying over a tree! {EMOJI['tree']}"
        elif outcome == "obstacle":
            self._drone_x, self._drone_y = nx, ny
            reward += cfg["r_obstacle"]
            msg = f"Hit obstacle! {EMOJI['obstacle']}"
        else:
            self._drone_x, self._drone_y = nx, ny
            if direction == "WAIT":
                reward = cfg.get("r_wait", -0.1)
                msg = "Drone is idling (WAIT)... 🔋"
            elif not msg:
                # Get cell type at current position
                current_cell = self._grid[self._drone_y][self._drone_x]
                if current_cell == "road":
                    msg = f"On road {EMOJI['road']}"
                else:
                    msg = f"Flying over {current_cell} {EMOJI.get(current_cell, '')}"

        # Move finalized, now check new distance
        _, dist_after = self._next_target()
        if dist_after is not None and dist_before is not None and outcome not in ["tree", "building", "ok"]:
            # Reward for moving closer, small penalty for moving away
            delta = dist_before - dist_after
            reward += delta * 0.05

        # Delivery check
        for i, (tx, ty) in enumerate(self._deliveries):
            if not self._delivered[i] and self._drone_x == tx and self._drone_y == ty:
                self._delivered[i] = True
                reward += cfg["r_delivery"]
                self._state.deliveries_done += 1
                msg = f"✅ Delivery {self._state.deliveries_done}/{len(self._deliveries)} done!"
                break

        # Battery drain
        self._battery = drain_battery(self._battery, cfg["battery_cost"])
        self._state.step_count += 1
        self._state.reward_total += reward
        bat_norm = max(0.0, float(self._battery) / cfg["battery_max"])
        self._state.battery = bat_norm
        self._state.drone_x = self._drone_x
        self._state.drone_y = self._drone_y

        self._step_records.append(dict(
            step=self._state.step_count,
            x=self._drone_x, y=self._drone_y,
            action=direction,
            reward=float(round(reward, 5)),
            battery=float(round(bat_norm, 4)),
        ))
        self._state.path_history = self._step_records

        # Done conditions
        done = False
        if all(self._delivered):
            done = True; msg = "🎉 All deliveries complete!"
        elif self._battery <= 0:
            reward += cfg["r_battery_dead"]
            self._state.reward_total += cfg["r_battery_dead"]
            done = True; msg = "🔋 Battery dead!"
        elif self._state.step_count >= cfg["max_steps"]:
            done = True; msg = "⏰ Max steps reached."

        self._state.done = done
        if done:
            self._persist_episode()

        return self._build_obs(reward, msg)

    @property
    def state(self) -> DroneState:
        return self._state

    def _persist_episode(self):
        try:
            cfg = self._cfg
            record_episode(
                task=self._state.task_name,
                steps=list(self._step_records),
                grid_meta=dict(
                    width=cfg["width"], height=cfg["height"]
                ),
                delivery_positions=[[d[0], d[1]] for d in self._deliveries],
                deliveries_done=self._state.deliveries_done,
                total_reward=float(round(self._state.reward_total, 4)),
            )
        except Exception as e:
            print(f"[DroneEnv] Episode record error: {e}")

    def _build_obs(self, reward: float, message: str) -> DroneObservation:
        cfg = self._cfg
        target, dist = self._next_target()
        return DroneObservation(
            grid=self._render_grid(),
            grid_width=int(cfg["width"]), grid_height=int(cfg["height"]),
            drone_x=self._drone_x, drone_y=self._drone_y,
            battery=float(max(0.0, float(self._battery) / cfg["battery_max"])),
            battery_steps_remaining=int(self._battery),
            deliveries_total=int(len(self._deliveries)),
            deliveries_done=int(self._state.deliveries_done),
            current_target=target,
            distance_to_target=float(dist) if dist is not None else None,
            step_count=int(self._state.step_count),
            max_steps=int(cfg["max_steps"]),
            reward_last=float(round(reward, 4)),
            reward_total=float(round(self._state.reward_total, 4)),
            score=float(round(GRADERS[self._state.task_name](self._state) * 100, 2)),
            done=bool(self._state.done),
            message=str(message),
            legend=dict(LEGEND),
            cell_types=list(self._grid),
        )

    def _render_grid(self) -> List[str]:
        rows = []
        for y, row in enumerate(self._grid):
            line = ""
            for x, cell in enumerate(row):
                if x == self._drone_x and y == self._drone_y:
                    line += EMOJI["drone"]
                    continue
                
                # Check for delivery targets
                is_delivery = False
                for i, (tx, ty) in enumerate(self._deliveries):
                    if x == tx and y == ty:
                        line += EMOJI["done_del"] if self._delivered[i] else EMOJI["delivery"]
                        is_delivery = True
                        break
                
                if not is_delivery:
                    # Map the internal symbol to emoji
                    # Now it looks up from EMOJI dict using the symbolic string in 'cell'
                    line += EMOJI.get(cell, cell)
            rows.append(line)
        return rows

    def _next_target(self):
        for i, (tx, ty) in enumerate(self._deliveries):
            if not self._delivered[i]:
                dist = float(abs(self._drone_x - tx) + abs(self._drone_y - ty))
                return (int(tx), int(ty)), float(dist)
        return None, None
