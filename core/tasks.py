"""
drone_env/core/tasks.py
Mission configurations for Drone Delivery missions.
"""

TASK_CONFIG = {
    "easy_delivery": {
        "width": 10,
        "height": 10,
        "n_buildings": 4,
        "n_trees": 4,
        "n_obstacles": 3,
        "n_deliveries": 1,
        "max_steps": 60,
        "battery_max": 60,
        "battery_cost": 1,
        "r_delivery": 1.0,
        "r_step": -0.05,
        "r_wait": -0.1,
        "r_obstacle": -0.1,
        "r_building": -0.1,  # Penalty for flying over buildings
        "r_tree": -0.1,      # Penalty for flying over trees
        "r_battery_dead": -0.5,
        "r_wall": -0.2,
        "r_blocked": -0.2,
    },
    "medium_delivery": {
        "width": 14,
        "height": 14,
        "n_buildings": 8,
        "n_trees": 6,
        "n_obstacles": 6,
        "n_deliveries": 3,
        "max_steps": 100,
        "battery_max": 100,
        "battery_cost": 1,
        "r_delivery": 0.8,
        "r_step": -0.05,
        "r_wait": -0.1,
        "r_obstacle": -0.15,
        "r_building": -0.1,
        "r_tree": -0.1,
        "r_battery_dead": -0.5,
        "r_wall": -0.2,
        "r_blocked": -0.2,
    },
    "hard_delivery": {
        "width": 18,
        "height": 18,
        "n_buildings": 12,
        "n_trees": 10,
        "n_obstacles": 10,
        "n_deliveries": 5,
        "max_steps": 160,
        "battery_max": 160,
        "battery_cost": 1,
        "r_delivery": 0.6,
        "r_step": -0.05,
        "r_wait": -0.1,
        "r_obstacle": -0.2,
        "r_building": -0.1,
        "r_tree": -0.1,
        "r_battery_dead": -1.0,
        "r_wall": -0.2,
        "r_blocked": -0.2,
    }
}
