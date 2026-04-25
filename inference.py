"""
SkyRelic Drone Delivery: Standardized Inference Script (v0.3.2)
===================================================
MANDATORY
- CLI supports --task and --steps for targeted testing.
- STDOUT FORMAT: [START], [STEP], [END]
- Logic: High-fidelity LLM reasoning with heuristic fallback.
"""

import asyncio
import os
import textwrap
import json
import sys
import argparse
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from openai import OpenAI

# Unified Imports - Canonical Package Paths with local fallbacks
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from models import DroneAction, DroneObservation
    from server.grid_world_environment import DroneDeliveryEnvironment
    from rl.trainer import get_action_from_policy
except ImportError:
    from drone_env.models import DroneAction, DroneObservation
    from drone_env.server.grid_world_environment import DroneDeliveryEnvironment
    from drone_env.rl.trainer import get_action_from_policy

# --- Load .env file for local development ---
def load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

load_dotenv()

# --- Configuration ------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY_KEY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

BENCHMARK = "drone_env"
ALL_TASKS = ["easy_delivery", "medium_delivery", "hard_delivery"]
TEMPERATURE = 0.7

# --- Structured output schema ------------------------------------------------

class NavigationAction(BaseModel):
    """Structured navigation action output from the LLM."""
    reasoning: str = Field(description="Brief explanation of why this direction was chosen")
    direction: str = Field(description="Movement direction: UP, DOWN, LEFT, RIGHT, or WAIT")

# --- Prompts ------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a drone navigation AI. Your goal is to deliver packages to targets in a grid world.
    
    Grid Mechanics:
    - (x, y) coordinates: x increases right, y increases down.
    - UP: y decreases
    - DOWN: y increases
    - LEFT: x decreases
    - RIGHT: x increases
    
    Constraints:
    - Avoid buildings and obstacles.
    - Battery drains per move.
    
    You MUST respond with a valid JSON object:
    {"reasoning": "<brief explanation>", "direction": "UP|DOWN|LEFT|RIGHT|WAIT"}
""").strip()

def build_user_prompt(obs: DroneObservation) -> str:
    return textwrap.dedent(f"""
        Pos: ({obs.drone_x}, {obs.drone_y})
        Battery: {obs.battery:.2f}
        Target: {obs.current_target}
        Distance: {obs.distance_to_target:.1f}
        Status: {obs.message}
        
        Plan your next move to reach the target efficiently.
    """).strip()

# --- Logging Helpers ---------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} final_score={score:.3f}", flush=True)

# --- Agent Logic -------------------------------------------------------------

def get_action(client: OpenAI, obs: DroneObservation) -> NavigationAction:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            max_tokens=100
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        action = NavigationAction(
            reasoning=data.get("reasoning", ""),
            direction=data.get("direction", "WAIT").upper(),
        )
        if action.direction not in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
            action.direction = "WAIT"
        return action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}. Activating heuristic fallback...", flush=True)
        heuristic_actions = get_action_from_policy(obs)
        fallback_dir = heuristic_actions.get(0, "WAIT")
        return NavigationAction(reasoning="Heuristic Fallback", direction=fallback_dir)

# --- Run Loop ----------------------------------------------------------------

async def run_task(task_id: str, env: DroneDeliveryEnvironment, client: OpenAI, step_limit: int = None) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(DroneAction(task_name=task_id))
        max_steps = step_limit if step_limit else (int(obs.max_steps) if obs.max_steps else 60)

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            # AI Decision
            nav_action = get_action(client, obs)
            action_str = nav_action.direction
            
            # Application
            if action_str == "WAIT":
                h_actions = get_action_from_policy(obs)
                obs = env.step(DroneAction(actions=h_actions))
            else:
                obs = env.step(DroneAction(direction=action_str))
            
            reward = float(obs.reward_last)
            done = bool(obs.done)
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score = float(obs.score) if obs.score is not None else 0.01
        success = (obs.deliveries_done == obs.deliveries_total) and obs.deliveries_total > 0

    except Exception as e:
        print(f"[DEBUG] run_task({task_id}) error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
        print(f"\n{'='*50}", flush=True)
        print(f"  Task        : {task_id}", flush=True)
        print(f"  Total Steps : {steps_taken}", flush=True)
        print(f"  Final Score : {score:.3f}", flush=True)
        print(f"  Success     : {success}", flush=True)
        print(f"{'='*50}\n", flush=True)

    return score

async def main() -> None:
    parser = argparse.ArgumentParser(description="SkyRelic Inference Engine")
    parser.add_argument("--task", type=str, default="all", choices=["easy_delivery", "medium_delivery", "hard_delivery", "all"])
    parser.add_argument("--steps", type=int, default=0, help="Override max steps for the mission")
    args = parser.parse_args()

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DroneDeliveryEnvironment()

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    step_limit = args.steps if args.steps > 0 else None

    for task_id in tasks_to_run:
        await run_task(task_id, env, client, step_limit)

if __name__ == "__main__":
    asyncio.run(main())
