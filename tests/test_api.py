"""
drone_env/tests/test_api.py
# Integration tests for the Drone Delivery Env FastAPI server.
 endpoints using TestClient.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from fastapi.testclient import TestClient
from drone_env.server.app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_root():
    """Test dashboard serving or JSON fallback."""
    r = client.get("/")
    assert r.status_code == 200
    ctype = r.headers.get("Content-Type", "")
    if "text/html" in ctype:
        assert "Drone Delivery" in r.text or "<!DOCTYPE html>" in r.text
    elif "application/json" in ctype:
        data = r.json()
        assert "name" in data
    else:
        # If no specific type, try to verify some content exists
        assert len(r.text) > 0


def test_tasks_list():
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    names = [t["name"] for t in tasks]
    assert "easy_delivery" in names
    assert "medium_delivery" in names
    assert "hard_delivery" in names


def test_reset_default():
    r = client.post("/reset")
    assert r.status_code == 200
    obs = r.json()
    assert "grid" in obs
    assert obs["done"] is False
    assert obs["step_count"] == 0
    assert 0 < obs["grid_width"] <= 20
    assert 0 < obs["grid_height"] <= 20


def test_reset_with_task():
    for task in ["easy_delivery", "medium_delivery", "hard_delivery"]:
        r = client.post("/reset", json={"task_name": task})
        assert r.status_code == 200
        obs = r.json()
        assert obs["done"] is False


def test_step_returns_observation():
    client.post("/reset", json={"task_name": "easy_delivery"})
    for direction in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
        r = client.post("/step", json={"direction": direction})
        assert r.status_code == 200
        obs = r.json()
        assert "reward_last" in obs
        assert "battery" in obs
        assert "done" in obs


def test_state_endpoint():
    client.post("/reset", json={"task_name": "easy_delivery"})
    r = client.get("/state")
    assert r.status_code == 200
    state = r.json()
    assert "episode_id" in state
    assert "task_name" in state
    assert state["task_name"] == "easy_delivery"


def test_grade_known_task():
    client.post("/reset", json={"task_name": "easy_delivery"})
    r = client.get("/grade/easy_delivery")
    assert r.status_code == 200
    score = r.json()["score"]
    assert 0.0 <= score <= 1.0


def test_grade_unknown_task():
    r = client.get("/grade/unknown_task")
    assert r.status_code == 404


def test_path_history_after_steps():
    client.post("/reset", json={"task_name": "easy_delivery"})
    for _ in range(5):
        client.post("/step", json={"direction": "RIGHT"})
    r = client.get("/path_history")
    assert r.status_code == 200
    history = r.json()["path_history"]
    assert len(history) == 5


def test_reward_in_range():
    """All per-step rewards must stay reasonable (no NaN, no extreme values)."""
    client.post("/reset", json={"task_name": "easy_delivery"})
    for direction in ["UP","RIGHT","DOWN","LEFT","WAIT","UP","RIGHT"]:
        r = client.post("/step", json={"direction": direction})
        obs = r.json()
        reward = obs["reward_last"]
        assert -2.0 <= reward <= 2.0, f"Reward out of range: {reward}"


def test_battery_decreases():
    client.post("/reset", json={"task_name": "easy_delivery"})
    r1 = client.post("/step", json={"direction": "WAIT"})
    bat1 = r1.json()["battery"]
    r2 = client.post("/step", json={"direction": "WAIT"})
    bat2 = r2.json()["battery"]
    assert bat2 <= bat1


def test_analyse_endpoint():
    r = client.get("/analyse/easy_delivery")
    assert r.status_code == 200
    data = r.json()
    # PathLearner returns a dict with "status" and "message" or stats
    assert "status" in data


def test_analyse_unknown_task():
    r = client.get("/analyse/nonexistent")
    assert r.status_code == 404


def test_full_episode_doesnt_crash():
    """Run a full episode of random steps — environment must not raise."""
    import random
    directions = ["UP","DOWN","LEFT","RIGHT","WAIT"]
    client.post("/reset", json={"task_name": "easy_delivery"})
    for _ in range(160):  # beyond max_steps to trigger done
        r = client.post("/step", json={"direction": random.choice(directions)})
        assert r.status_code == 200
        if r.json().get("done"):
            break
    # Should be able to reset again cleanly
    r = client.post("/reset", json={"task_name": "easy_delivery"})
    assert r.status_code == 200
    assert r.json()["done"] is False
