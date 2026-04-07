"""
SkyRelic Drone Delivery: Standardized Inference Script
===================================================
MANDATORY
- Environment variables: HF_TOKEN, API_BASE_URL, MODEL_NAME
- STDOUT FORMAT: [START], [STEP], [END]
- Participants must use OpenAI Client for all LLM calls.
"""

import asyncio
import os
import textwrap
import sys
from typing import List, Optional
from pathlib import Path

from openai import OpenAI

# Local Project Imports
ROOT_DIR = Path(__file__).parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Load .env for local development convenience
def load_dotenv(path: Path):
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")
load_dotenv(ROOT_DIR / ".env")

# Use the established Drone Environment
from drone_env.server.grid_world_environment import DroneDeliveryEnvironment
from drone_env.models import DroneAction, DroneObservation

# --- Configuration ------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
# Use a placeholder if no key is found to prevent initialization crash during local tests
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY_KEY"

# Defaults are set only for API_BASE_URL and MODEL_NAME as per requirements
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"

TASK_NAME = os.getenv("DRONE_TASK", "drone_env.core.graders:grade_easy")
BENCHMARK = os.getenv("DRONE_BENCHMARK", "drone_env_v1")

MAX_STEPS = 60
TEMPERATURE = 0.0
MAX_TOKENS = 50
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a drone navigation AI. Your goal is to deliver packages to targets.
    Each turn, you receive the drone's position, battery, target, and distance.
    Valid Actions: UP, DOWN, LEFT, RIGHT, WAIT.
    Respond with exactly one action name in uppercase. No quotes.
    """
).strip()

# --- Logging Helpers ---------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Format reward to 2 decimal places as per requirement
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Format rewards to 2 decimal places as per requirement
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- Agent Logic -------------------------------------------------------------
def build_user_prompt(obs: DroneObservation) -> str:
    return textwrap.dedent(
        f"""
        Pos: ({obs.drone_x}, {obs.drone_y})
        Battery: {obs.battery:.2f}
        Target: {obs.current_target}
        Distance: {obs.distance_to_target:.1f}
        Status: {obs.message}
        Available Actions: UP, DOWN, LEFT, RIGHT, WAIT
        """
    ).strip()

def get_model_action(client: OpenAI, obs: DroneObservation) -> str:
    user_prompt = build_user_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip().upper()
        if text not in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
            return "WAIT"
        return text
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return "WAIT"

# --- Run Loop ----------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize environment locally
    # Note: the sample template uses docker integration, but for local 
    # hackathon development, we initialize the DroneDeliveryEnvironment directly.
    env = DroneDeliveryEnvironment()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.010
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Initial reset
        obs = env.reset(DroneAction(task_name=TASK_NAME))
        
        # Max steps from environment or local constant
        max_limit = int(obs.max_steps) if obs.max_steps else MAX_STEPS

        for step in range(1, max_limit + 1):
            if obs.done:
                break

            action_str = get_model_action(client, obs)

            # Step environment
            obs = env.step(DroneAction(direction=action_str))

            reward = float(obs.reward_last)
            done = bool(obs.done)
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Final score (already normalized by the environment's grader [0.01, 0.99])
        score = float(obs.score) if obs.score is not None else 0.010
        success = (obs.deliveries_done == obs.deliveries_total) and obs.deliveries_total > 0

    finally:
        # Mandatory close and log emission
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
