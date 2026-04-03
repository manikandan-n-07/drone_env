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

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from drone_env.rl.model import PathQNet, ACTIONS, ACTION2IDX, CELL2IDX
from drone_env.server.grid_world_environment import DroneDeliveryEnvironment
from drone_env.models import DroneAction

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
def obs_to_tensors(obs, device):
    """Converts a DroneObservation to the tensors required by PathQNet."""
    # 1. Grid (H*W)
    # Using cell_types instead of emojis for better robustness
    grid = obs.cell_types
    flat_grid = []
    for row in grid:
        for cell in row:
            # Map emoji back to int (simplified)
            idx = CELL2IDX.get(cell, 0) # default road
            flat_grid.append(idx)
    grid_t = torch.tensor([flat_grid], dtype=torch.long, device=device)

    # 2. Drone XY (normalized)
    drone_xy = torch.tensor([[obs.drone_x / obs.grid_width, obs.drone_y / obs.grid_height]], dtype=torch.float, device=device)

    # 3. Battery (0-1)
    battery = torch.tensor([[obs.battery]], dtype=torch.float, device=device)

    # 4. Target XY (normalized)
    if obs.current_target:
        tx, ty = obs.current_target
        target_xy = torch.tensor([[tx / obs.grid_width, ty / obs.grid_height]], dtype=torch.float, device=device)
    else:
        target_xy = torch.tensor([[0.0, 0.0]], dtype=torch.float, device=device)

    return grid_t, drone_xy, battery, target_xy

# ── Training Loop ─────────────────────────────────────────────────────────────
def train(task_name: str, episodes: int, device_name: str):
    device = torch.device(device_name)
    print(f"Starting training for {task_name} on {device}...")

    os.makedirs("data", exist_ok=True)
    with open("data/train.log", "a") as f:
        f.write(f"\n>>> Neural Training Engine Started [TASK: {task_name}]\n")
        
    env = DroneDeliveryEnvironment()
    policy_net = PathQNet(embed_dim=64).to(device)
    target_net = PathQNet(embed_dim=64).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(REPLAY_SIZE)
    epsilon = EPS_START

    # Automatically resume if model exists
    task_short = task_name.split('_')[0]
    model_path = f"data/{task_short}.pth"
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
        state_t = obs_to_tensors(obs, device)
        total_reward = 0
        done = False

        while not done:
            # Select Action
            if random.random() < epsilon:
                action_idx = random.randint(0, len(ACTIONS) - 1)
            else:
                with torch.no_grad():
                    q_values = policy_net(*state_t)
                    action_idx = q_values.argmax().item()

            action_str = ACTIONS[action_idx]
            
            # Step
            next_obs = env.step(DroneAction(direction=action_str))
            reward = next_obs.reward_last
            done = next_obs.done
            total_reward += reward

            next_state_t = obs_to_tensors(next_obs, device)
            
            # Save to buffer
            memory.push(state_t, action_idx, reward, next_state_t, done)
            state_t = next_state_t

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
        
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if ep % 50 == 0:
            log_msg = f"Episode {ep}/{episodes} | Avg Reward: {total_reward:.2f} | Epsilon: {epsilon:.2f}"
            print(log_msg)
            torch.save(policy_net.state_dict(), model_path)
            with open("data/train.log", "a") as f:
                f.write(log_msg + " (Periodic Save)\n")

    # Save
    task_short = task_name.split('_')[0] 
    model_path = f"data/{task_short}.pth"
    torch.save(policy_net.state_dict(), model_path)
    
    msg = f"Training complete. Model saved to {model_path}."
    print(msg)
    with open("data/train.log", "a") as f:
        f.write(msg + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="easy_delivery")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    train(args.task, args.episodes, device)
