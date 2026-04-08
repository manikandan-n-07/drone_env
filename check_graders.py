import yaml
import importlib
import sys
import os
from pathlib import Path

def check_graders():
    # Ensure current directory is in path
    sys.path.insert(0, os.getcwd())
    
    yaml_path = Path("openenv.yaml")
    if not yaml_path.exists():
        print("❌ Error: openenv.yaml not found!")
        return

    try:
        with open(yaml_path, 'r') as f:
            spec = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error parsing openenv.yaml: {e}")
        return

    tasks = spec.get("tasks", [])
    print(f"🔍 Found {len(tasks)} tasks in openenv.yaml\n")

    valid_count = 0

    for task in tasks:
        task_id = task.get("id")
        grader_str = task.get("grader")
        
        print(f"--- Task: {task_id} ---")
        if not grader_str:
            print("  ❌ No grader field defined.")
            continue
        
        print(f"  Attempting to resolve: {grader_str}")
        try:
            if ":" not in grader_str:
                print(f"  ❌ Invalid format. Expected 'module:function'")
                continue
            
            module_path, func_name = grader_str.split(":")
            
            module = None
            # Try full path first (as it will be in the platform)
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                # Local fallback: try without 'drone_env.' prefix if we're in the project root
                if module_path.startswith("drone_env."):
                    alt_path = module_path.replace("drone_env.", "", 1)
                    try:
                        module = importlib.import_module(alt_path)
                    except ImportError as eb:
                        print(f"  ❌ Module search failed (tried {module_path} and {alt_path}): {eb}")
            
            if module:
                func = getattr(module, func_name, None)
                if func and callable(func):
                    print(f"  ✅ SUCCESS: Found {func_name} in {module.__name__}")
                    valid_count += 1
                else:
                    print(f"  ❌ {func_name} not found or not callable in {module.__name__}")
            else:
                pass # Already printed error
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\nSummary: {valid_count} valid graders found.")
    if valid_count >= 3:
        print("🚀 LOCAL CHECK PASSED.")
    else:
        print("🚨 LOCAL CHECK FAILED.")

if __name__ == "__main__":
    check_graders()
