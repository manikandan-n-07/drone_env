"""
rl/unsloth_trainer.py
Fine-tuning Drone Navigation Agents using Unsloth LoRA.
This script converts RL-collected experiences into Instruction-Tuning data.
"""
import json
import os
import torch
from pathlib import Path
from typing import List, Dict
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Disable W&B logging to prevent interactive prompts in Colab
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Configuration ---
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
OUTPUT_DIR = "outputs/drone_fleet_adapter"

# --- Data Preparation ---

def get_reasoning(drone_pos, target_pos, action):
    """Generates a simple reasoning string for the drone's action."""
    dx = target_pos[0] - drone_pos[0]
    dy = target_pos[1] - drone_pos[1]
    
    reasoning = f"The target is {'East' if dx > 0 else 'West' if dx < 0 else 'aligned'} and {'South' if dy > 0 else 'North' if dy < 0 else 'aligned'}."
    return f"Reasoning: {reasoning} Action: {action}"

def load_and_format_data(tasks: List[str] = ["easy", "medium", "hard"]) -> List[Dict]:
    """Loads trajectories from memory.json and formats them for ChatML/SFT."""
    all_conversations = []
    
    for task in tasks:
        mem_path = Path(f"data/{task}/memory.json")
        if not mem_path.exists():
            continue
            
        with open(mem_path, "r") as f:
            episodes = json.load(f)
            
        print(f"[Load] Found {len(episodes)} episodes for {task}")
        
        for ep in episodes:
            if ep.get("deliveries_done", 0) == 0 and ep.get("total_reward", 0) < 1.0:
                continue
                
            grid_w, grid_h = ep["grid_meta"]["width"], ep["grid_meta"]["height"]
            
            for step_data in ep["steps"]:
                for drone in step_data["drones"]:
                    # Get target (simplified for first target)
                    target = ep["delivery_positions"][0] if ep["delivery_positions"] else (0,0)
                    
                    state_desc = (
                        f"Environment: {task} delivery task on {grid_w}x{grid_h} grid.\n"
                        f"Current Position: ({drone['x']}, {drone['y']})\n"
                        f"Target: {target}\n"
                        f"Battery: {drone['battery']:.2f}\n"
                        f"Step: {step_data['step']}"
                    )
                    
                    # Generate Reasoning-enhanced response
                    response = get_reasoning((drone['x'], drone['y']), target, drone['action'])
                    
                    prompt = (
                        "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\nPredict the next optimal navigation action and explain your reasoning.\n\n"
                        f"### Input:\n{state_desc}\n\n"
                        f"### Response:\n{response}"
                    )
                    
                    all_conversations.append({"text": prompt})
                    
    return all_conversations

# --- Training Loop ---

def train():
    # 1. Load Model
    print(">>> Initializing Unsloth Model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        load_in_4bit = LOAD_IN_4BIT,
        device_map = "auto"
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, 
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 3. Load Dataset
    data = load_and_format_data()
    if not data:
        print("!!! Error: No training data found in data/*/memory.json. Run train.py first!")
        return
        
    dataset = Dataset.from_list(data)
    print(f">>> Prepared {len(dataset)} training samples.")

    # 4. SFT Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 4,
            warmup_steps = 10,
            max_steps = 100,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = f"outputs/{args.task}", # Task-specific runs
        ),
    )

    # 5. Execute Training
    print(">>> Starting Fine-Tuning...")
    trainer.train()

    # 6. Save Adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f">>> Training Complete. Adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drone LLM Finetuning")
    parser.add_argument("--task", type=str, default="fleet", help="Suffix for the adapter (easy, medium, hard, fleet)")
    args = parser.parse_args()
    
    # Update global output dir structure
    global OUTPUT_DIR
    OUTPUT_DIR = f"outputs/{args.task}/adapter"
    
    train()
