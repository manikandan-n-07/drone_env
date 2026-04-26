"""
train.py
Standalone advanced training script for Drone Delivery RL.
Usage: python train.py --task easy_delivery --episodes 1000
"""
import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rl.model import PathQNet, ACTIONS, ACTION2IDX, CELL2IDX
from rl.trainer import record_episode
from server.grid_world_environment import DroneDeliveryEnvironment
from models import DroneAction

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA = 0.99
BATCH_SIZE = 64
LR = 1e-4
REPLAY_SIZE = 10000
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995

# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# ── Helper to process observations ────────────────────────────────────────────
def obs_to_tensors(obs, drone_idx, device):
    """Converts a DroneObservation for a specific drone to the tensors required by PathQNet."""
    # 1. Grid (H*W)
    grid = obs.cell_types
    flat_grid = []
    for row in grid:
        for cell in row:
            idx = CELL2IDX.get(cell, 0)
            flat_grid.append(idx)
    grid_t = torch.tensor([flat_grid], dtype=torch.long, device=device)

    # 2. Specific Drone XY (normalized)
    drone = obs.drones[drone_idx]
    drone_xy = torch.tensor([[drone.x / obs.grid_width, drone.y / obs.grid_height]], dtype=torch.float, device=device)

    # 3. Battery (0-1)
    battery = torch.tensor([[drone.battery]], dtype=torch.float, device=device)

    # 4. Target XY (normalized)
    # Get target coordinates for THIS specific drone
    target_xy_val = [0.0, 0.0]
    if drone.target_id is not None and drone.target_id < len(obs.targets):
        tx, ty = obs.targets[drone.target_id]
        target_xy_val = [tx / obs.grid_width, ty / obs.grid_height]
    
    target_xy = torch.tensor([target_xy_val], dtype=torch.float, device=device)

    return grid_t, drone_xy, battery, target_xy

# ── Training Loop ─────────────────────────────────────────────────────────────
def train(task_name: str, episodes: int, device_name: str, run_unsloth: bool = False):
    device = torch.device(device_name)
    print(f"Starting training for {task_name} on {device}...")

    task_short = task_name.split('_')[0]
    task_dir = f"data/{task_short}"
    os.makedirs(task_dir, exist_ok=True)
    task_log = f"{task_dir}/train.log"

    with open(task_log, "a") as f:
        f.write(f"\n>>> Neural Training Engine Started [TASK: {task_name}]\n")
        
    env = DroneDeliveryEnvironment()
    policy_net = PathQNet(embed_dim=64).to(device)
    target_net = PathQNet(embed_dim=64).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(REPLAY_SIZE)
    epsilon = EPS_START

    # Tracking metrics
    episode_rewards = []
    training_losses = []
    avg_reward_history = []

    # Automatically resume if model exists
    task_short = task_name.split('_')[0]
    model_dir = Path("data") / task_short
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"
    csv_path = model_dir / "train_metrics.csv"
    
    # Initialize or Fix CSV Headers
    header = "episode,reward,moving_avg_reward,epsilon,loss,deliveries,steps\n"
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write(header)
    else:
        # Check if first line is the header, if not, prepend it
        with open(csv_path, "r") as f:
            first_line = f.readline()
        if first_line != header:
            with open(csv_path, "r") as f:
                content = f.read()
            with open(csv_path, "w") as f:
                f.write(header + content)
            
    if os.path.exists(model_path):
        print(f"Loading existing weights from {model_path}...")
        try:
            policy_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            target_net.load_state_dict(policy_net.state_dict())
            epsilon = EPS_END # Start with low exploration if resuming
            with open("data/train.log", "a") as f:
                f.write(f"Resuming from checkpoint: {model_path}\n")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}. Starting fresh.")

    for ep in range(episodes):
        obs = env.reset(DroneAction(task_name=task_name))
        total_reward = 0
        done = False
        # Episode loop
        episode_steps = []
        step_count = 0
        
        max_ep_steps = 250 if "hard" in task_name else (150 if "medium" in task_name else 100)
        
        while not done and step_count < max_ep_steps:
            step_count += 1
            
            fleet_actions = {}
            fleet_action_indices = {}
            fleet_states_t = {}

            # 1. Action selection for EACH drone in the fleet
            for i, drone in enumerate(obs.drones):
                state_t = obs_to_tensors(obs, i, device)
                fleet_states_t[drone.id] = state_t
                
                # Epsilon-greedy
                if random.random() < epsilon:
                    action_idx = random.randint(0, len(ACTIONS) - 1)
                else:
                    with torch.no_grad():
                        q_values = policy_net(*state_t)
                        action_idx = q_values.argmax().item()
                
                fleet_action_indices[drone.id] = action_idx
                fleet_actions[drone.id] = ACTIONS[action_idx]

            # 2. Record step data for visualization
            step_data = {
                "step": step_count,
                "drones": []
            }
            for d in obs.drones:
                step_data["drones"].append({
                    "id": d.id,
                    "x": d.x,
                    "y": d.y,
                    "battery": d.battery,
                    "action": fleet_actions.get(d.id, "WAIT")
                })
            episode_steps.append(step_data)

            # 3. Collaborative Step
            next_obs = env.step(DroneAction(actions=fleet_actions))
            reward = next_obs.reward_last # Average team reward
            done = next_obs.done
            total_reward += reward

            # 4. Save experiences to buffer (One sample per drone)
            for i, drone in enumerate(obs.drones):
                next_state_t = obs_to_tensors(next_obs, i, device)
                memory.push(fleet_states_t[drone.id], fleet_action_indices[drone.id], reward, next_state_t, done)
            
            obs = next_obs

            # Optimize
            if len(memory) > BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                
                # Unzip batch
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Batch processing
                def cat_tensors(t_list):
                    return [torch.cat([t[i] for t in t_list]) for i in range(4)]
                
                b_grid, b_drone, b_bat, b_target = cat_tensors(states)
                bn_grid, bn_drone, bn_bat, bn_target = cat_tensors(next_states)
                
                b_actions = torch.tensor(actions, device=device).unsqueeze(1)
                b_rewards = torch.tensor(rewards, device=device, dtype=torch.float).unsqueeze(1)
                b_dones   = torch.tensor(dones, device=device, dtype=torch.float).unsqueeze(1)
                
                # 1. Current Q values
                current_q = policy_net(b_grid, b_drone, b_bat, b_target).gather(1, b_actions)
                
                # 2. Next Q values from target net
                with torch.no_grad():
                    next_q = target_net(bn_grid, bn_drone, bn_bat, bn_target).max(1)[0].unsqueeze(1)
                    expected_q = b_rewards + (GAMMA * next_q * (1 - b_dones))
                
                # 3. Loss & Step
                loss = nn.MSELoss()(current_q, expected_q)
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                training_losses.append(loss.item())

        # Record episode results to memory.json
        try:
            record_episode(
                task=task_name,
                steps=episode_steps, 
                grid_meta=dict(width=env._cfg["width"], height=env._cfg["height"]),
                delivery_positions=[list(p) for p in env._deliveries],
                deliveries_done=int(next_obs.deliveries_done),
                total_reward=float(round(total_reward, 4))
            )
        except Exception as e:
            print(f"[DEBUG] Failed to record episode: {e}")
            pass
        
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Save metrics to CSV for EVERY episode
        avg_loss = sum(training_losses[-10:]) / 10 if len(training_losses) >= 10 else 0
        avg_reward_curve = sum(avg_reward_history[-10:]) / 10 if len(avg_reward_history) >= 10 else total_reward
        with open(csv_path, "a") as f:
            f.write(f"{ep},{total_reward:.4f},{avg_reward_curve:.4f},{epsilon:.4f},{avg_loss:.6f},{next_obs.deliveries_done},{step_count}\n")

        if ep % 50 == 0:
            log_msg = f"Episode {ep}/{episodes} | Avg Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}"
            print(log_msg)
            torch.save(policy_net.state_dict(), model_path)
            with open(task_log, "a") as f:
                f.write(log_msg + " (Periodic Save)\n")
        
        episode_rewards.append(total_reward)
        avg_reward_history.append(sum(episode_rewards[-10:]) / min(len(episode_rewards), 10))

    # Save
    # Save final model
    task_short = task_name.split('_')[0] 
    model_dir = f"data/{task_short}"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/model.pth"
    torch.save(policy_net.state_dict(), model_path)
    
    msg = f"Training complete. Model saved to {model_path}."
    print(msg)
    with open(task_log, "a") as f:
        f.write(msg + "\n")

    # 13. End Training
    print(f"\n>>> Training Complete. Best Avg Reward: {max(avg_reward_history) if avg_reward_history else 0:.2f}")

    # 14. Optional: Run Unsloth Fine-tuning
    if run_unsloth:
        print("\n>>> Starting Unsloth LLM Fine-tuning Stage...")
        try:
            import subprocess
            subprocess.run([sys.executable, "rl/unsloth_trainer.py"], check=True)
        except Exception as e:
            print(f"!!! Unsloth training failed: {e}")

    # Generate Evidence Plots
    task_short = task_name.split('_')[0]
    result_dir = f"results/{task_short}"
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. Reward Curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    plt.plot(avg_reward_history, color='red', label='Moving Avg (10)')
    plt.title(f"Reward Curve - {task_name}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/reward_curve.png")
    plt.close()

    # 2. Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, color='orange')
    plt.title(f"Loss Curve - {task_name}")
    plt.xlabel("Training Step")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    # Use log scale if losses are high
    if len(training_losses) > 0 and max(training_losses) > 1:
        plt.yscale('log')
    plt.savefig(f"{result_dir}/loss_curve.png")
    plt.close()
    
    print(f"Evidence plots saved to {result_dir}/reward_curve.png and {result_dir}/loss_curve.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="easy_delivery", help="Task name (easy_delivery, medium_delivery, hard_delivery) or 'all'")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes (overrides defaults)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--unsloth", action="store_true", help="Run Unsloth LLM fine-tuning after RL training")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    
    # Task to episodes mapping
    TASK_DEFAULTS = {
        "easy_delivery": 500,
        "medium_delivery": 1000,
        "hard_delivery": 2000
    }

    if args.task == "all":
        tasks_to_run = ["easy_delivery", "medium_delivery", "hard_delivery"]
    else:
        tasks_to_run = [args.task]

    for task in tasks_to_run:
        eps = args.episodes if args.episodes is not None else TASK_DEFAULTS.get(task, 100)
        print(f"\n{'='*60}\nENQUEUING TASK: {task} | TARGET: {eps} EPISODES\n{'='*60}\n")
        train(task, eps, device, args.unsloth)
