"""
server/app.py
FastAPI server for the Drone Delivery OpenEnv — modular version.
"""
from __future__ import annotations
import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import logging
from collections import deque
import time
import json

# Add root to sys.path for local imports
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Unified Imports - Canonical Package Paths
from drone_env.models import DroneAction, DroneObservation, DroneState
from drone_env.server.grid_world_environment import DroneDeliveryEnvironment
from drone_env.core.tasks import TASK_CONFIG
from drone_env.core.graders import GRADERS
from drone_env.rl.trainer import PathLearner, get_action_from_policy

app = FastAPI(
    title="Drone Delivery OpenEnv",
    description="Real-world drone delivery RL environment.",
    version="0.2.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- TERMINAL LOG CAPTURE ---
class TerminalLogManager:
    def __init__(self, capacity=100):
        self.buffer = deque(maxlen=capacity)
    
    def add_log(self, msg: str):
        self.buffer.append(msg)

terminal_log_manager = TerminalLogManager()
terminal_log_manager.add_log(f"SYSTEM: Drone Delivery Env Terminal Log Engine Initialized v{app.version}")
terminal_log_manager.add_log(f"SYSTEM: Waiting for neural link on port 8000...")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Filter out polling noise
    path = request.url.path
    if path in ["/logs", "/terminal_logs", "/health", "/rewards", "/events"]:
        return await call_next(request)
        
    start_time = time.time()
    response = await call_next(request)
    duration = (time.time() - start_time) * 1000
    
    log_line = f"{request.client.host} - \"{request.method} {path} HTTP/1.1\" {response.status_code} ({duration:.1f}ms)"
    terminal_log_manager.add_log(log_line)
    return response
# ----------------------------

# Serve Frontend
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_env = DroneDeliveryEnvironment()

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.2.1"}

@app.post("/reset", response_model=DroneObservation)
async def reset(action: Optional[DroneAction] = None):
    return _env.reset(action)

@app.post("/step", response_model=DroneObservation)
async def step(action: DroneAction):
    return _env.step(action)

@app.get("/state", response_model=DroneState)
async def get_state():
    return _env.state

@app.get("/grade/{task_name}")
async def grade(task_name: str):
    if task_name not in GRADERS:
        raise HTTPException(404, detail=f"Unknown task: {task_name}")
    return {"task": task_name, "score": float(GRADERS[task_name](_env.state))}

@app.get("/analyse/{task_name}")
async def analyse(task_name: str):
    if task_name not in TASK_CONFIG:
         raise HTTPException(404, detail=f"Unknown task: {task_name}")
    return PathLearner.analyse_episodes(task_name)

@app.get("/path_history")
async def path_history():
    return {"path_history": _env.state.path_history}

@app.get("/tasks")
async def list_tasks():
    return {"tasks": [{"name": k, **v} for k, v in TASK_CONFIG.items()]}

@app.get("/graders")
async def list_graders():
    return {"graders": list(GRADERS.keys())}

@app.get("/logs")
async def get_logs():
    log_path = BASE_DIR / "data" / "train.log"
    if not log_path.exists():
        return {"logs": []}
    with open(log_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return {"logs": ["> Waiting for Neural Stream... (run train.py to begin)"]}
    return {"logs": lines[-50:]} 

@app.get("/terminal_logs")
async def get_terminal_logs():
    return {"logs": list(terminal_log_manager.buffer)}

@app.get("/rewards")
async def get_rewards():
    """Return the reward configuration for the current task."""
    task_name = _env.state.task_name or "drone_env.core.graders:grade_easy"
    config = TASK_CONFIG.get(task_name, {})
    # Filter only reward keys
    rewards = {k: v for k, v in config.items() if k.startswith("r_")}
    return {"task": task_name, "rewards": rewards}

@app.get("/events")
async def get_events():
    """Return recent significant reward events."""
    history = _env.state.path_history
    # Filter for delivery/collision/failure events
    events = []
    for entry in history[-20:]: # Check last 20 steps
        msg = entry.get("message", "")
        # Events have high rewards or exclamation marks or emoji targets
        if entry.get("reward", 0) > 0.05 or "!" in msg or "🎉" in msg or "✅" in msg:
            events.append(entry)
    return {"events": events[-10:]}

@app.get("/memory_logs")
async def get_memory_logs():
    memory_path = BASE_DIR / "data" / "memory.json"
    if not memory_path.exists():
        return {"episodes": []}
    try:
        with open(memory_path, "r") as f:
            data = json.load(f)
            summary = []
            for ep in data[-5:]:
                summary.append({
                    "task": ep.get("task", "unknown"),
                    "reward": ep.get("total_reward", 0),
                    "steps": ep.get("total_steps", 0),
                    "deliveries": ep.get("deliveries_done", 0)
                })
            return {"episodes": summary[::-1]}
    except Exception as e:
        return {"episodes": [], "error": str(e)}

@app.post("/predict")
async def predict(obs: DroneObservation):
    task_name = _env.state.task_name or "drone_env.core.graders:grade_easy"
    action_str = get_action_from_policy(obs, task_name)
    return {"direction": action_str}

@app.get("/")
async def root():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"name": "Drone Delivery OpenEnv", "ui": "/ui"}

@app.get("/favicon.png")
async def favicon_png():
    icon_path = BASE_DIR / "src" / "img" / "icon.png"
    if icon_path.exists():
        headers = {"Cache-Control": "no-cache, no-store, must-revalidate"}
        return FileResponse(icon_path, media_type="image/png", headers=headers)
    raise HTTPException(404, detail="Icon not found.")

@app.get("/favicon.ico")
async def favicon_ico():
    return await favicon_png()

@app.get("/ui")
async def ui():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(404, detail="UI index.html not found.")

def main():
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    print(f"Starting Drone Delivery Server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
