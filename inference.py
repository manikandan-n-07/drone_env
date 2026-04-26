"""
SkyRelic Drone Delivery: Standardized Inference Script (v0.4.0)
===================================================
- CLI: --task [easy_delivery|medium_delivery|hard_delivery|all], --use_llm [api|local|none]
- Logic: Dual support for API-based reasoning and Unsloth Local Adapters.
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

# --- Unsloth Support ---
try:
    from unsloth import FastLanguageModel
    import torch
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# Unified Imports - Canonical Package Paths
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

# --- DOTENV Loader ---
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

# --- Globals for Local Model ---
LOCAL_MODEL = None
LOCAL_TOKENIZER = None

# --- Configuration ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY_KEY"
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

BENCHMARK = "drone_env"
ALL_TASKS = ["easy_delivery", "medium_delivery", "hard_delivery"]
TEMPERATURE = 0.1

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Predict the next optimal navigation action and explain your reasoning.

### Input:
{input}

### Response:
"""

# --- Structured output schema ---

class NavigationAction(BaseModel):
    """Structured navigation action output."""
    reasoning: str = Field(description="Explanation of why this direction was chosen")
    direction: str = Field(description="Movement direction: UP, DOWN, LEFT, RIGHT, or WAIT")

# --- Logging Helpers ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float) -> None:
    print(f"[END] success={str(success).lower()} steps={steps} final_score={score:.3f}", flush=True)

# --- Agent Logic ---

def get_api_action(client: OpenAI, obs: DroneObservation) -> NavigationAction:
    """Fetches action from an external LLM API."""
    system_prompt = "You are a drone AI. Respond with JSON: {\"reasoning\": \"...\", \"direction\": \"UP|DOWN|LEFT|RIGHT|WAIT\"}"
    user_prompt = f"Pos: ({obs.drone_x}, {obs.drone_y}) Battery: {obs.battery:.2f} Target: {obs.current_target} Dist: {obs.distance_to_target:.1f}"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
            max_tokens=100
        )
        data = json.loads(completion.choices[0].message.content or "{}")
        return NavigationAction(
            reasoning=data.get("reasoning", "API Decision"),
            direction=data.get("direction", "WAIT").upper()
        )
    except Exception:
        return NavigationAction(reasoning="API Error Fallback", direction="WAIT")

def get_local_action(obs: DroneObservation, task_id: str) -> NavigationAction:
    """Inference for the fine-tuned Unsloth model."""
    global LOCAL_MODEL, LOCAL_TOKENIZER
    if not LOCAL_MODEL:
        return NavigationAction(reasoning="Model Not Loaded", direction="WAIT")

    # Match state_desc format from unsloth_trainer.py
    state_desc = (
        f"Environment: {task_id.replace('_delivery', '')} task.\n"
        f"Current Position: ({obs.drone_x}, {obs.drone_y})\n"
        f"Battery: {obs.battery:.2f}\n"
        f"Distance to Target: {obs.distance_to_target:.1f}\n"
        f"Status: {obs.message}"
    )
    
    inputs = LOCAL_TOKENIZER([ALPACA_PROMPT.format(input=state_desc)], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = LOCAL_MODEL.generate(**inputs, max_new_tokens=32, temperature=0.1)
    
    decoded = LOCAL_TOKENIZER.batch_decode(outputs, skip_special_tokens=True)[0]
    raw_response = decoded.split("### Response:")[-1].strip()
    
    # Extract Reasoning and Action
    reasoning_part = "Decision generated by fine-tuned model."
    action_part = "WAIT"
    
    if "Action:" in raw_response:
        parts = raw_response.split("Action:")
        reasoning_part = parts[0].replace("Reasoning:", "").strip()
        action_part = parts[1].strip().upper()
    else:
        # Fallback keyword search
        for move in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
            if move in raw_response.upper():
                action_part = move
                break
            
    return NavigationAction(reasoning=reasoning_part, direction=action_part)

# --- Run Loop ---

async def run_task(task_id: str, env: DroneDeliveryEnvironment, client: OpenAI, mode: str, step_limit: int = None) -> float:
    steps_total = 0
    score = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=mode.upper())

    try:
        obs = env.reset(DroneAction(task_name=task_id))
        max_steps = step_limit if step_limit else 60

        for step in range(1, max_steps + 1):
            if obs.done: break

            # Action Selection
            if mode == "api":
                nav_action = get_api_action(client, obs)
            elif mode == "local":
                nav_action = get_local_action(obs, task_id)
            else:
                h_actions = get_action_from_policy(obs)
                nav_action = NavigationAction(reasoning="Heuristic Mode", direction=h_actions.get(0, "WAIT"))

            # Fallback for WAIT (use heuristic if LLM stalls)
            if nav_action.direction == "WAIT" or nav_action.direction == "":
                h_actions = get_action_from_policy(obs)
                action_str = h_actions.get(0, "WAIT")
                obs = env.step(DroneAction(actions=h_actions))
            else:
                action_str = nav_action.direction
                obs = env.step(DroneAction(direction=action_str))
            
            steps_total = step
            log_step(step=step, action=action_str, reward=float(obs.reward_last), done=bool(obs.done), error=None)
            if obs.done: break

        score = float(obs.score) if obs.score is not None else 0.01
        success = (obs.deliveries_done == obs.deliveries_total) and obs.deliveries_total > 0

    except Exception as e:
        print(f"[DEBUG] Error: {e}")
    finally:
        log_end(success=success, steps=steps_total, score=score)

    return score

# --- Main ---

async def main() -> None:
    parser = argparse.ArgumentParser(description="SkyRelic Inference Engine")
    parser.add_argument("--task", type=str, default="all", choices=["easy_delivery", "medium_delivery", "hard_delivery", "all"])
    parser.add_argument("--use_llm", type=str, default="api", choices=["api", "local", "none"])
    parser.add_argument("--adapter_path", type=str, default="AUTO")
    parser.add_argument("--steps", type=int, default=0)
    args = parser.parse_args()

    # Automatic Path Resolution
    if args.adapter_path == "AUTO":
        level = args.task.replace("_delivery", "") if args.task != "all" else "fleet"
        args.adapter_path = f"./outputs/{level}/adapter"

    # Disable HF Hub warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = DroneDeliveryEnvironment()

    # Load Local Model if needed
    if args.use_llm == "local":
        global LOCAL_MODEL, LOCAL_TOKENIZER
        if not UNSLOTH_AVAILABLE:
            print("!!! Error: unsloth not installed. Cannot use local mode.")
            return
        
        # Check if path exists
        if not os.path.exists(args.adapter_path):
            print(f"!!! Error: Adapter not found at {args.adapter_path}. Run training with '--task {level}' first!")
            return

        print(f">>> Loading Fine-tuned Adapter from {args.adapter_path}...")
        LOCAL_MODEL, LOCAL_TOKENIZER = FastLanguageModel.from_pretrained(
            model_name = args.adapter_path,
            max_seq_length = 2048,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(LOCAL_MODEL)

    tasks_to_run = ALL_TASKS if args.task == "all" else [args.task]
    for task_id in tasks_to_run:
        await run_task(task_id, env, client, args.use_llm, args.steps if args.steps > 0 else None)

if __name__ == "__main__":
    asyncio.run(main())
