"""
drone_env/client.py
Python SDK client for the Drone Delivery Env.

Usage:
    from drone_env.client import DroneEnvClient

    client = DroneEnvClient("http://localhost:8000")
    obs = client.reset("easy_delivery")
    obs = client.step("UP")
    state = client.state()
    score = client.grade("easy_delivery")
    analytics = client.analyse("easy_delivery")
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import httpx


class DroneEnvClient:
    """
    Thin HTTP client wrapping all /reset /step /state /grade /analyse
    endpoints of the Drone Delivery OpenEnv server.
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ── Core OpenEnv API ──────────────────────────────────────────────────────

    def reset(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """Reset the environment. Optionally pass task_name to choose difficulty."""
        payload: Dict[str, Any] = {}
        if task_name:
            payload["task_name"] = task_name
        r = self._http.post("/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, direction: str) -> Dict[str, Any]:
        """
        Take one step.
        direction: 'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'WAIT'
        """
        r = self._http.post("/step", json={"direction": direction.upper()})
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        """Return the full current environment state."""
        r = self._http.get("/state")
        r.raise_for_status()
        return r.json()

    def grade(self, task_name: str) -> float:
        """Return the current grade (0.0–1.0) for the given task."""
        r = self._http.get(f"/grade/{task_name}")
        r.raise_for_status()
        return float(r.json().get("score", 0.0))

    # ── Analytics ─────────────────────────────────────────────────────────────

    def analyse(self, task_name: str) -> Dict[str, Any]:
        """Return RL analytics from stored memory.json for a task."""
        r = self._http.get(f"/analyse/{task_name}")
        r.raise_for_status()
        return r.json()

    def path_history(self) -> list:
        """Return the step-by-step path history of the current episode."""
        r = self._http.get("/path_history")
        r.raise_for_status()
        return r.json().get("path_history", [])

    # ── Meta ──────────────────────────────────────────────────────────────────

    def tasks(self) -> list:
        """List available tasks and their configs."""
        r = self._http.get("/tasks")
        r.raise_for_status()
        return r.json().get("tasks", [])

    def health(self) -> Dict[str, Any]:
        """Ping the server health endpoint."""
        r = self._http.get("/health")
        r.raise_for_status()
        return r.json()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._http.close()

    def close(self):
        self._http.close()

    # ── Convenience: run full episode ─────────────────────────────────────────

    def run_random_episode(
        self,
        task_name: str = "easy_delivery",
        max_steps: int = 200,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a full episode with random actions. Returns final state + score.
        Useful for smoke-testing the environment.
        """
        import random
        directions = ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]

        obs = self.reset(task_name)
        if verbose:
            print(f"[DroneEnvClient] Starting {task_name} — "
                  f"{obs['deliveries_total']} deliveries on a "
                  f"{obs['grid_width']}×{obs['grid_height']} grid")

        for step in range(1, max_steps + 1):
            if obs.get("done"):
                break
            action = random.choice(directions)
            obs = self.step(action)
            if verbose and (step % 20 == 0 or obs.get("done")):
                print(f"  step={step:3d}  action={action:<5}  "
                      f"reward={obs['reward_last']:+.3f}  "
                      f"total={obs['reward_total']:+.3f}  "
                      f"battery={obs['battery']*100:.0f}%  "
                      f"deliveries={obs['deliveries_done']}/{obs['deliveries_total']}  "
                      f"done={obs['done']}")

        score = self.grade(task_name)
        if verbose:
            print(f"[DroneEnvClient] Episode done — score={score:.4f}  "
                  f"msg='{obs.get('message', '')}'")
        return {"observation": obs, "score": score}


# ── Quick smoke test when run directly ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    print(f"Connecting to {url} …")
    with DroneEnvClient(url) as c:
        print("Health:", json.dumps(c.health(), indent=2))
        print("Tasks: ", json.dumps(c.tasks(), indent=2))
        result = c.run_random_episode("easy_delivery", verbose=True)
        print(f"\nFinal score: {result['score']:.4f}")
        analytics = c.analyse("easy_delivery")
        print("\nAnalytics:", json.dumps(analytics, indent=2))
