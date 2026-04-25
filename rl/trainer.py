"""
drone_env/rl/trainer.py
# Experience replay and training logic for the Drone Delivery Env.
"""
import json
import os
import torch
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
try:
    from drone_env.models import DroneObservation
except ImportError:
    from models import DroneObservation

DATA_DIR = Path("data")


def record_episode(
    task: str,
    steps: List[Dict],
    grid_meta: Dict,
    delivery_positions: List[List[int]],
    deliveries_done: int,
    total_reward: float
):
    """
    Persist an episode's path history and metadata to memory.json.
    """
    task_short = task.split('_')[0]
    task_dir = DATA_DIR / task_short
    task_dir.mkdir(parents=True, exist_ok=True)
    memory_path = task_dir / "memory.json"
        
    episodes = []
    if memory_path.exists():
        try:
            with open(memory_path, "r") as f:
                episodes = json.load(f)
        except (json.JSONDecodeError, ValueError):
            episodes = []
            
    episode_data = {
        "task": task,
        "steps": steps,
        "grid_meta": grid_meta,
        "delivery_positions": delivery_positions,
        "deliveries_done": deliveries_done,
        "total_reward": total_reward,
        "total_steps": len(steps)
    }
    
    episodes.append(episode_data)
    
    # Keep last 100 episodes
    if len(episodes) > 100:
        episodes = episodes[-100:]
        
    with open(memory_path, "w") as f:
        json.dump(episodes, f, indent=2)


class PathLearner:
    """
    Analyzes episode data stored in memory.json.
    """
    
    @staticmethod
    def analyse_episodes(task_name: str):
        """
        Computes statistics for the specified task from memory.json.
        """
        task_short = task_name.split('_')[0]
        memory_path = DATA_DIR / task_short / "memory.json"
        
        if not memory_path.exists():
            return {"status": "No data", "message": f"Collect data for {task_name} first!"}
            
        try:
            with open(memory_path, "r") as f:
                episodes = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {"status": "Error", "message": "Invalid memory file."}
            
        # Filter by task
        task_episodes = [e for e in episodes if e["task"] == task_name]
        
        if not task_episodes:
            return {"status": "No data", "message": f"No episodes for {task_name}"}
            
        total_ep = len(task_episodes)
        
        # Ensure total_reward and others are strictly in (0.01, 0.99) even for old data
        avg_reward = sum(max(0.01, min(0.99, e.get("total_reward", 0.0))) for e in task_episodes) / total_ep
        avg_steps = sum(e.get("total_steps", 0) for e in task_episodes) / total_ep
        avg_del = sum(e.get("deliveries_done", 0) for e in task_episodes) / total_ep
        
        # Action distribution
        action_counts = {"UP": 0, "DOWN": 0, "LEFT": 0, "RIGHT": 0, "WAIT": 0}
        for ep in task_episodes:
            for step in ep["steps"]:
                act = step.get("action", "WAIT").upper()
                if act in action_counts:
                    action_counts[act] += 1
                        
        return {
            "status": "Success",
            "total_episodes": total_ep,
            "avg_reward": float(max(0.01, min(0.99, round(avg_reward, 3)))),
            "avg_steps": round(avg_steps, 1),
            "avg_deliveries": f"{avg_del:.1f}",
            "action_distribution": action_counts
        }


# --- PyTorch Integration ------------------------------------------------------

from .model import PathQNet, CELL2IDX, ACTIONS

def get_action_from_policy(obs: DroneObservation, task_name: str = "easy_delivery") -> Dict[int, str]:
    """
    Autonomous mode: Predict next move for each drone using a greedy heuristic with obstacle avoidance.
    """
    actions = {}
    godown_x, godown_y = (obs.grid_width // 2, 0)
    
    occupied_next = set() # (nx, ny)

    # Helper to check if a move is blocked
    def is_blocked(nx, ny):
        if not (0 <= nx < obs.grid_width and 0 <= ny < obs.grid_height):
            return True
        if (nx, ny) in occupied_next:
            return True
        if not obs.cell_types or ny >= len(obs.cell_types) or nx >= len(obs.cell_types[0]):
            return False
        cell = obs.cell_types[ny][nx]
        return cell == "obstacle"

    for drone in obs.drones:
        # Determine destination
        if drone.has_package:
            tx, ty = godown_x, godown_y
        elif drone.target_id is not None and drone.target_id < len(obs.targets):
            tx, ty = obs.targets[drone.target_id]
        else:
            tx, ty = godown_x, godown_y
        
        dx, dy = tx - drone.x, ty - drone.y
        if dx == 0 and dy == 0:
            actions[drone.id] = "WAIT"
            continue

        # Try directions in order of preference (best for Manhattan)
        possible_moves = [] # list of (priority_score, move_name, next_x, next_y)
        
        # RIGHT/LEFT
        if dx != 0:
            possible_moves.append((abs(dx), "RIGHT" if dx > 0 else "LEFT", drone.x + (1 if dx > 0 else -1), drone.y))
        # DOWN/UP
        if dy != 0:
            possible_moves.append((abs(dy), "DOWN" if dy > 0 else "UP", drone.x, drone.y + (1 if dy > 0 else -1)))
            
        # Add orthogonal moves with lower priority if primary moves are blocked
        if dx == 0: # Already at target X, try X moves as backup
            possible_moves.append((0, "RIGHT", drone.x + 1, drone.y))
            possible_moves.append((0, "LEFT", drone.x - 1, drone.y))
        if dy == 0: # Already at target Y, try Y moves as backup
            possible_moves.append((0, "DOWN", drone.x, drone.y + 1))
            possible_moves.append((0, "UP", drone.x, drone.y - 1))

        # Sort by priority (Manhattan advantage first)
        possible_moves.sort(key=lambda x: x[0], reverse=True)
        
        final_move = "WAIT"
        for _, name, nx, ny in possible_moves:
            if not is_blocked(nx, ny):
                final_move = name
                occupied_next.add((nx, ny))
                break
                
        actions[drone.id] = final_move
    
    print(f"[Policy] Actions generated: {actions}")
    return actions
