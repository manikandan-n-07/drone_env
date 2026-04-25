"""
drone_delivery_env/core/grid_generator.py
Procedural generation of connected city maps for drone delivery.
"""
from typing import List, Tuple, Dict
import torch

EMOJI = {
    "drone": "🚁",
    "road": "🛣️",
    "building": "🏢",
    "tree": "🌳",
    "obstacle": "🚧",
    "delivery": "📦",
    "done_del": "✅",
    "godown": "🏭"
}

LEGEND = {
    "🚁": "Drone",
    "🛣️": "Road (Safe)",
    "🏢": "Building (Penalty)",
    "🌳": "Tree (Penalty)",
    "🚧": "Obstacle (Penalty)",
    "📦": "Delivery Target",
    "✅": "Delivered",
    "🏭": "Godown (Warehouse)"
}


def generate_city_map(cfg: Dict, rng: torch.Generator) -> Tuple[List[List[str]], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Generate a connected grid of roads, buildings, trees, and delivery targets.
    Returns (grid, deliveries, godown_pos)
    """
    W, H = cfg["width"], cfg["height"]
    grid = [["road" for _ in range(W)] for _ in range(H)]
    
    # Place Godown at the top-center
    godown_pos = (W // 2, 0)
    grid[godown_pos[1]][godown_pos[0]] = "godown"

    # Simple placement of buildings and trees
    def place_random(symbol: str, count: int):
        placed = 0
        while placed < count:
            rx = torch.randint(0, W, (1,), generator=rng).item()
            ry = torch.randint(0, H, (1,), generator=rng).item()
            if grid[ry][rx] == "road":
                grid[ry][rx] = symbol
                placed += 1

    place_random("building", cfg["n_buildings"])
    place_random("tree", cfg["n_trees"])
    place_random("obstacle", cfg["n_obstacles"])
    
    # Place deliveries
    deliveries = []
    placed_d = 0
    while placed_d < cfg["n_deliveries"]:
        rx = torch.randint(0, W, (1,), generator=rng).item()
        ry = torch.randint(1, H, (1,), generator=rng).item() # Avoid top row for deliveries
        if grid[ry][rx] == "road":
            grid[ry][rx] = "delivery"
            deliveries.append((rx, ry))
            placed_d += 1
            
    return grid, deliveries, godown_pos
