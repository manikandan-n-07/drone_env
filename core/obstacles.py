"""
drone_delivery_env/core/obstacles.py
Collision detection and movement validation for the drone.
"""
from typing import List, Tuple


def check_move(grid: List[List[str]], x: int, y: int, width: int, height: int) -> Tuple[str, str]:
    """
    Validate a move to (x, y). Returns:
    (outcome, cell_type)
    outcome: ok | wall | blocked | building | tree | obstacle
    """
    if not (0 <= x < width and 0 <= y < height):
        return "wall", ""
    
    cell = grid[y][x]
    if cell == "building":
        return "building", "building"
    if cell == "tree":
        return "tree", "tree"
    if cell == "obstacle":
        return "obstacle", "obstacle"
    
    return "ok", cell
