"""
drone_env/tests/test_env.py
# Unit tests for the Drone Delivery Env core logic and outcomes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import torch
from drone_env.core.grid_generator import generate_city_map
from drone_env.core.tasks import TASK_CONFIG
from drone_env.core.obstacles import check_move
from drone_env.core.drone import compute_next_pos, drain_battery
from drone_env.core.graders import grade_easy, grade_medium, grade_hard
from drone_env.models import DroneState


# ── Grid generator ────────────────────────────────────────────────────────────

def test_grid_shape():
    cfg = TASK_CONFIG["easy_delivery"]
    rng = torch.Generator(); rng.manual_seed(42)
    grid, deliveries, start = generate_city_map(cfg, rng)
    assert len(grid) == cfg["height"]
    assert all(len(row) == cfg["width"] for row in grid)
    assert len(deliveries) == cfg["n_deliveries"]
    assert 0 <= start[0] < cfg["width"]
    assert 0 <= start[1] < cfg["height"]


def test_start_on_road():
    cfg = TASK_CONFIG["easy_delivery"]
    rng = torch.Generator(); rng.manual_seed(7)
    grid, _, start = generate_city_map(cfg, rng)
    sx, sy = start
    assert grid[sy][sx] == "road"


def test_delivery_cells_present():
    cfg = TASK_CONFIG["medium_delivery"]
    rng = torch.Generator(); rng.manual_seed(99)
    grid, deliveries, _ = generate_city_map(cfg, rng)
    for dx, dy in deliveries:
        assert grid[dy][dx] == "delivery"


# ── Obstacle helper ───────────────────────────────────────────────────────────

def _make_grid(rows):
    return rows

def test_check_move_wall():
    grid = [["road"]*5 for _ in range(5)]
    outcome, cell = check_move(grid, -1, 0, 5, 5)
    assert outcome == "wall"

def test_check_move_blocked():
    grid = [["building","road"], ["road","road"]]
    outcome, cell = check_move(grid, 0, 0, 2, 2)
    assert outcome == "building"
    assert cell == "building"

def test_check_move_obstacle():
    grid = [["road","obstacle"], ["road","road"]]
    outcome, cell = check_move(grid, 1, 0, 2, 2)
    assert outcome == "obstacle"

def test_check_move_ok():
    grid = [["road","road"], ["road","road"]]
    outcome, cell = check_move(grid, 1, 1, 2, 2)
    assert outcome == "ok"


# ── Drone movement ────────────────────────────────────────────────────────────

def test_compute_next_pos_up():
    assert compute_next_pos(3, 3, "UP") == (3, 2)

def test_compute_next_pos_down():
    assert compute_next_pos(3, 3, "DOWN") == (3, 4)

def test_compute_next_pos_left():
    assert compute_next_pos(3, 3, "LEFT") == (2, 3)

def test_compute_next_pos_right():
    assert compute_next_pos(3, 3, "RIGHT") == (4, 3)

def test_compute_next_pos_wait():
    assert compute_next_pos(3, 3, "WAIT") == (3, 3)

def test_drain_battery():
    assert drain_battery(100, 1) == 99
    assert drain_battery(1, 5) == 0  # clamps at 0


# ── Graders ───────────────────────────────────────────────────────────────────

def _make_state(**kw):
    defaults = dict(
        deliveries_done=0, deliveries_total=2,
        step_count=0, battery=1.0
    )
    defaults.update(kw)
    return DroneState(**defaults)

def test_grade_zero_deliveries():
    s = _make_state(deliveries_done=0, deliveries_total=2, step_count=10, battery=0.8)
    score = grade_easy(s)
    assert 0.01 <= score <= 0.99
    assert score < 0.5  # no deliveries → low score

def test_grade_all_deliveries():
    s = _make_state(deliveries_done=2, deliveries_total=2, step_count=50, battery=0.9)
    score = grade_easy(s)
    assert 0.7 <= score <= 0.99  # all deliveries → high score

def test_grade_clamped():
    s = _make_state(deliveries_done=2, deliveries_total=2, step_count=1, battery=1.0)
    score = grade_easy(s)
    assert 0.01 <= score <= 0.99

def test_grade_medium_hard_bounds():
    s = _make_state(deliveries_done=4, deliveries_total=4, step_count=100, battery=0.7)
    assert 0.01 <= grade_medium(s) <= 0.99
    assert 0.01 <= grade_hard(s) <= 0.99


# ── Task configs ──────────────────────────────────────────────────────────────

def test_all_tasks_present():
    assert "easy_delivery" in TASK_CONFIG
    assert "medium_delivery" in TASK_CONFIG
    assert "hard_delivery" in TASK_CONFIG

def test_task_has_required_keys():
    required = ["width","height","n_buildings","n_trees","n_obstacles",
                "n_deliveries","max_steps","battery_max","battery_cost",
                "r_delivery","r_step","r_obstacle","r_battery_dead","r_wall","r_blocked"]
    for task, cfg in TASK_CONFIG.items():
        for key in required:
            assert key in cfg, f"{task} missing key: {key}"

def test_task_difficulty_ordering():
    easy   = TASK_CONFIG["easy_delivery"]
    medium = TASK_CONFIG["medium_delivery"]
    hard   = TASK_CONFIG["hard_delivery"]
    assert easy["width"] < medium["width"] < hard["width"]
    assert easy["n_deliveries"] < medium["n_deliveries"] < hard["n_deliveries"]
    assert easy["max_steps"] < medium["max_steps"] < hard["max_steps"]
