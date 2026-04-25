import os
import sys

# Add the project root to sys.path
project_root = r"c:\Users\megaa\Downloads\drone_env"
if project_root not in sys.path:
    sys.path.append(project_root)

from server.grid_world_environment import DroneDeliveryEnvironment
from models import DroneAction

def test_tree_flyover():
    env = DroneDeliveryEnvironment()
    obs = env.reset()
    
    # Find a tree on the grid
    tree_pos = None
    for y, row in enumerate(env._grid):
        for x, cell in enumerate(row):
            if cell == "🌳":
                tree_pos = (x, y)
                break
        if tree_pos:
            break
            
    if not tree_pos:
        print("No tree found on small grid, retrying reset...")
        for _ in range(10):
            obs = env.reset()
            for y, row in enumerate(env._grid):
                for x, cell in enumerate(row):
                    if cell == "🌳":
                        tree_pos = (x, y)
                        break
                if tree_pos: break
            if tree_pos: break

    if not tree_pos:
        print("Still no tree found. Test aborted.")
        return

    print(f"Drone start: ({env._drone_x}, {env._drone_y})")
    print(f"Tree found at: {tree_pos}")

    # Move the drone to be adjacent to the tree if possible, or just teleport for testing
    # Actually, we can just simulate the step logic or manually set position and then step
    
    # Let's find direction to the tree
    tx, ty = tree_pos
    dx, dy = env._drone_x, env._drone_y
    
    # Teleport to adjacent
    if tx > 0:
        env._drone_x = tx - 1
        env._drone_y = ty
        direction = "RIGHT"
    elif tx < env._cfg["width"] - 1:
        env._drone_x = tx + 1
        env._drone_y = ty
        direction = "LEFT"
    elif ty > 0:
        env._drone_x = tx
        env._drone_y = ty - 1
        direction = "DOWN"
    else:
        env._drone_x = tx
        env._drone_y = ty + 1
        direction = "UP"
        
    print(f"Teleported drone to adjacent: ({env._drone_x}, {env._drone_y})")
    
    # Perform move into tree
    action = DroneAction(direction=direction)
    obs = env.step(action)
    
    print(f"Action: {direction}")
    print(f"New Pos: ({env._drone_x}, {env._drone_y})")
    print(f"Reward: {obs.reward_last}")
    print(f"Message: {obs.message}")
    
    assert env._drone_x == tx and env._drone_y == ty, "Drone should have moved onto the tree cell"
    assert obs.reward_last == -0.1, f"Reward should be -0.1, got {obs.reward_last}"
    print("Test passed! Flying over tree works and reward is exactly -0.1")

if __name__ == "__main__":
    test_tree_flyover()
