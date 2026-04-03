import torch
import random
from typing import List, Tuple

def generate_noise_map(width: int, height: int) -> torch.Tensor:
    # A simple single-layer 'generator' that creates a correlated noise map
    # representing roads vs buildings.
    noise = torch.rand((1, 1, height, width))
    # simple smoothing using average pooling to create clumps
    if height >= 3 and width >= 3:
        pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        noise = pool(noise)
    return noise[0, 0]

def generate_grid(difficulty: str) -> Tuple[List[List[str]], tuple, List[tuple], List[tuple]]:
    width, height = 12, 12
    noise_map = generate_noise_map(width, height)
    
    # Threshold for roads (lower values will be roads, higher building/tree)
    # Target approx 40% roads
    threshold = torch.quantile(noise_map, 0.4).item()
    
    grid = [["" for _ in range(width)] for _ in range(height)]
    road_coords = []
    
    for y in range(height):
        for x in range(width):
            val = noise_map[y, x].item()
            if val < threshold:
                grid[y][x] = "🛣️"
                road_coords.append((x, y))
            else:
                # randomly assign building or tree
                if random.random() < 0.6:
                    grid[y][x] = "🏢"
                else:
                    grid[y][x] = "🌳"
                    
    # Guarantee at least one road cell
    if not road_coords:
        grid[0][0] = "🛣️"
        road_coords.append((0, 0))
        
    # Get configuration based on difficulty
    if difficulty == "easy":
        num_targets = 2
        num_obstacles = 2
    elif difficulty == "medium":
        num_targets = 4
        num_obstacles = 4
    else:
        num_targets = 7
        num_obstacles = 6
        
    # We must restrict to available road coords minus 1 for drone
    available_coords = random.sample(road_coords, min(len(road_coords), num_targets + num_obstacles + 1))
    
    drone_start = available_coords.pop(0)
    grid[drone_start[1]][drone_start[0]] = "🚁"
    
    targets = []
    for _ in range(min(num_targets, len(available_coords))):
        t_coord = available_coords.pop(0)
        grid[t_coord[1]][t_coord[0]] = "🎯"
        targets.append(t_coord)
        
    obstacles = []
    for _ in range(min(num_obstacles, len(available_coords))):
        o_coord = available_coords.pop(0)
        grid[o_coord[1]][o_coord[0]] = "🚧"
        obstacles.append(o_coord)
        
    return grid, drone_start, targets, obstacles
