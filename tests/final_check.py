import requests
import time

BASE_URL = "http://localhost:8000"

def check_task(task_id):
    print(f"\n--- Testing Task: {task_id} ---")
    # 1. Reset
    r = requests.post(f"{BASE_URL}/reset", json={"task_name": task_id})
    if r.status_code != 200:
        print(f"❌ Reset failed: {r.status_code}")
        return False
    
    obs = r.json()
    print(f"✅ Reset: Grid={obs['grid_width']}x{obs['grid_height']}, Target={obs['current_target']}")
    
    # 2. Check Score Range (should be 0.01 - 0.99)
    # At start, score should be around 0.01-0.10
    score = obs['score']
    if 0.01 <= score <= 0.99:
        print(f"✅ Initial Score check: {score} (within 0.01-0.99)")
    else:
        print(f"❌ Initial Score out of range: {score}")
        return False
    
    # 3. Check Grader endpoint
    r = requests.get(f"{BASE_URL}/grade/{task_id}")
    if r.status_code != 200:
        print(f"❌ Grader endpoint failed: {r.status_code}")
        return False
    
    grade_res = r.json()
    print(f"✅ Grader score: {grade_res['score']} (matches obs: {grade_res['score'] == score})")
    return True

def run_checks():
    # Wait for server to be ready
    for _ in range(5):
        try:
            requests.get(f"{BASE_URL}/health")
            break
        except:
            time.sleep(2)
            
    # Check Graders discovery
    r = requests.get(f"{BASE_URL}/graders")
    graders = r.json().get('graders', [])
    print(f"\n--- Discovering Graders ---")
    print(f"✅ Found {len(graders)} graders: {graders}")
    
    expected = ["easy_delivery", "medium_delivery", "hard_delivery"]
    for t in expected:
        if t not in graders:
            print(f"❌ Missing task: {t}")
            return
            
    # Test each task
    for t in expected:
        if not check_task(t):
            print(f"\n❌ FINAL CHECK FAILED ON {t}")
            return
            
    print("\n" + "="*40)
    print("🏆 ALL CHECKS PASSED: Environment is GOLDEN")
    print("="*40)

if __name__ == "__main__":
    run_checks()
