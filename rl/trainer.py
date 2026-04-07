"""
drone_env/rl/trainer.py
# Experience replay and training logic for the Drone Delivery Env.
"""
import json
import os
import torch
from typing import List, Dict, Any
from pathlib import Path

MEMORY_PATH = Path("data/memory.json")


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
    if not os.path.exists("data"):
        os.makedirs("data")
        
    episodes = []
    if MEMORY_PATH.exists():
        try:
            with open(MEMORY_PATH, "r") as f:
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
        
    with open(MEMORY_PATH, "w") as f:
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
        if not MEMORY_PATH.exists():
            return {"status": "No data", "message": "Collect data first!"}
            
        try:
            with open(MEMORY_PATH, "r") as f:
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

def get_action_from_policy(obs: Any, task_name: str = "easy_delivery") -> str:
    """
    Autonomous mode: Predict next move using trained PyTorch model for the specific task.
    """
    task_short = task_name.split('_')[0]
    model_path = f"data/{task_short}.pth"

    try:
        if not os.path.exists(model_path):
            # FALLBACK: Greedy heuristic if model isn't trained yet
            if obs.current_target:
                tx, ty = obs.current_target
                dx, dy = tx - obs.drone_x, ty - obs.drone_y
                
                # Preferred directions based on distance
                if abs(dx) > abs(dy):
                    if dx > 0 and obs.drone_x < obs.grid_width - 1: return "RIGHT"
                    if dx < 0 and obs.drone_x > 0: return "LEFT"
                    if dy > 0 and obs.drone_y < obs.grid_height - 1: return "DOWN"
                    if dy < 0 and obs.drone_y > 0: return "UP"
                else:
                    if dy > 0 and obs.drone_y < obs.grid_height - 1: return "DOWN"
                    if dy < 0 and obs.drone_y > 0: return "UP"
                    if dx > 0 and obs.drone_x < obs.grid_width - 1: return "RIGHT"
                    if dx < 0 and obs.drone_x > 0: return "LEFT"
                
                return "WAIT"
            return "WAIT" 

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PathQNet(embed_dim=64).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        
        # Grid conversion for tensors
        grid = obs.cell_types
        flat_grid = []
        for row in grid:
            for cell in row:
                flat_grid.append(CELL2IDX.get(cell, 0))
        
        g_t = torch.tensor([flat_grid], dtype=torch.long, device=device)
        d_t = torch.tensor([[obs.drone_x/obs.grid_width, obs.drone_y/obs.grid_height]], dtype=torch.float, device=device)
        b_t = torch.tensor([[obs.battery]], dtype=torch.float, device=device)
        
        if obs.current_target:
            tx, ty = obs.current_target
            t_t = torch.tensor([[tx/obs.grid_width, ty/obs.grid_height]], dtype=torch.float, device=device)
        else:
            t_t = torch.tensor([[0.0, 0.0]], dtype=torch.float, device=device)

        with torch.no_grad():
            q_values = model(g_t, d_t, b_t, t_t)
            action_idx = q_values.argmax().item()
            action_str = ACTIONS[action_idx]
            
            # FINAL SAFETY CHECK: Model might be overconfident in hitting a wall
            if action_str == "LEFT" and obs.drone_x <= 0: action_str = "WAIT"
            if action_str == "RIGHT" and obs.drone_x >= obs.grid_width - 1: action_str = "WAIT"
            if action_str == "UP" and obs.drone_y <= 0: action_str = "WAIT"
            if action_str == "DOWN" and obs.drone_y >= obs.grid_height - 1: action_str = "WAIT"
            
            return action_str
            
    except Exception as e:
        print(f"[Trainer] Inference Error: {e}")
        return "WAIT"
