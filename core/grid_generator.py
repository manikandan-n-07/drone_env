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
    "done_del": "✅"
}

LEGEND = {
    "🚁": "Drone",
    "🛣️": "Road (Safe)",
    "🏢": "Building (Penalty)",
    "🌳": "Tree (Penalty)",
    "🚧": "Obstacle (Penalty)",
    "📦": "Delivery Target",
    "✅": "Delivered"
}


def generate_city_map(cfg: Dict, rng: torch.Generator) -> Tuple[List[List[str]], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Generate a connected grid of roads, buildings, trees, and delivery targets.
    Returns (grid, deliveries, start_pos)
    """
    W, H = cfg["width"], cfg["height"]
    # Internal grid uses symbolic strings: road, building, tree, obstacle, delivery
    grid = [["road" for _ in range(W)] for _ in range(H)]
    
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
        ry = torch.randint(0, H, (1,), generator=rng).item()
        if grid[ry][rx] == "road":
            grid[ry][rx] = "delivery"
            deliveries.append((rx, ry))
            placed_d += 1
            
    # Find start position on a road
    start_pos = (0, 0)
    found_start = False
    for _ in range(100):
        sx = torch.randint(0, W, (1,), generator=rng).item()
        sy = torch.randint(0, H, (1,), generator=rng).item()
        if grid[sy][sx] == "road":
            start_pos = (sx, sy)
            found_start = True
            break
            
    if not found_start:
        # Fallback to the first road found
        for y in range(H):
            for x in range(W):
                if grid[y][x] == "road":
                    start_pos = (x, y)
                    break
    
    return grid, deliveries, start_pos
