import os
import sys

# Add the project root to sys.path
project_root = r"c:\Users\megaa\Downloads\drone_env"
if project_root not in sys.path:
    sys.path.append(project_root)

from rl.trainer import get_action_from_policy
from models import DroneObservation

def test_boundary_avoidance():
    # Scenario: Drone is at the left boundary (x=0).
    # Target is at x=-5 (impossible, but simulates a target "beyond" the wall).
    # Heuristic should NOT return "LEFT".
    
    obs = DroneObservation(
        grid=[], grid_width=10, grid_height=10,
        drone_x=0, drone_y=5,
        battery=1.0, battery_steps_remaining=100,
        deliveries_total=1, deliveries_done=0,
        current_target=(-5, 5), # Target to the left of the left wall
        step_count=0, max_steps=100,
        reward_last=0, reward_total=0,
        done=False,
        cell_types=[["🛣️"]*10]*10
    )
    
    # We should ensure the model doesn't exist so it uses the fallback
    # The current trainer check is if data/easy.pth exists.
    
    action = get_action_from_policy(obs, "easy_delivery")
    print(f"Drone at (0,5), Target at (-5,5). Suggested action: {action}")
    
    assert action != "LEFT", "Should NOT suggest LEFT when at x=0"
    print("Test passed! Heuristic avoids the left wall.")

if __name__ == "__main__":
    test_boundary_avoidance()
