import asyncio
import os
import textwrap
from typing import List, Optional
import sys
from pathlib import Path

# Add core directories to sys.path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Manual .env loading helper (replaces python-dotenv for simplicity)
def load_dotenv(path: Path):
    if path.exists():
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"): continue
                if "=" not in line: continue
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
        except Exception as e:
            print(f"[DEBUG] .env load error: {e}", file=sys.stderr)

load_dotenv(ROOT_DIR / ".env")

from openai import OpenAI
from drone_env.server.grid_world_environment import DroneDeliveryEnvironment
from drone_env.models import DroneAction

# MANDATORY Environment Variables
# Defaults are set ONLY for API_BASE_URL and MODEL_NAME as per checklist
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# For local testing if no HF_TOKEN is provided but OPENAI_API_KEY is
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")

# Final API_KEY selection (Must not have a hardcoded default)
API_KEY = HF_TOKEN if HF_TOKEN else OPENAI_KEY

# Optional variables
DRONE_TASK = os.getenv("DRONE_TASK", "drone_env.core.graders:grade_easy")
BENCHMARK = os.getenv("DRONE_BENCHMARK", "drone_env_v1")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # Optional for Docker testing

# Hyperparameters
MAX_STEPS = 60
TEMPERATURE = 0.0 # Set to 0 for deterministic navigation with Qwen
MAX_TOKENS = 30   # Increased slightly for Qwen reasoning
SUCCESS_SCORE_THRESHOLD = 0.5  # 50% score for success

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a drone navigation AI. Your goal is to deliver all packages to their destinations.
    Each step, you will see the drone's position, battery level, current target position, and distance.
    Actions: UP, DOWN, LEFT, RIGHT, WAIT.
    Format: Respond with exactly ONE action name in uppercase.
    Example: UP
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # The format required by benchmarks
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(obs) -> str:
    return textwrap.dedent(
        f"""
        Pos: ({obs.drone_x}, {obs.drone_y})
        Battery: {obs.battery:.2f}
        Target: {obs.current_target}
        Distance: {obs.distance_to_target:.1f}
        Message: {obs.message}
        Available Actions: UP, DOWN, LEFT, RIGHT, WAIT
        Action?
        """
    ).strip()

def get_model_action(client: OpenAI, obs) -> str:
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
        action = (completion.choices[0].message.content or "").strip().upper()
        # Validation for allowed actions
        if action not in ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
            return "WAIT"
        return action
    except Exception as exc:
        # In case of API failure, log to stderr and return WAIT
        print(f"[DEBUG] Model request failed: {exc}", file=sys.stderr, flush=True)
        return "WAIT"

async def main() -> None:
    # Initialize OpenAI client according to benchmarks
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize our environment locally
    env = DroneDeliveryEnvironment()

    rewards: List[float] = []
    steps_taken = 0
    score_val = 0.0
    success = False

    log_start(task=DRONE_TASK, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Initial reset
        obs = env.reset(DroneAction(task_name=DRONE_TASK))
        
        # Use dynamic max_steps from environment config
        current_max_steps = int(obs.max_steps) if obs.max_steps else MAX_STEPS
        
        for step in range(1, current_max_steps + 1):
            if obs.done:
                break

            # Get action from model
            action_str = get_model_action(client, obs)

            # Execution in environment
            obs = env.step(DroneAction(direction=action_str))

            reward = obs.reward_last
            done = obs.done
            error = None # Error field as per benchmark (normally handled via exceptions)

            rewards.append(reward)
            steps_taken = step
            
            # Step logging
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        # Calculate final metrics
        score_val = float(obs.score)  # Graded score is already normalized to [0.01, 0.99]
        success = (obs.deliveries_done == obs.deliveries_total) if obs.deliveries_total > 0 else False

    except Exception as e:
        print(f"[DEBUG] Error during inference: {e}", file=sys.stderr, flush=True)
    finally:
        # Always output the [END] line
        log_end(success=success, steps=steps_taken, score=score_val, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
