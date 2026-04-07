import sys
from pathlib import Path
# Add the parent directory of this project to sys.path
sys.path.insert(0, str(Path.cwd().parent))

from drone_env.core.graders import compute_grade
from drone_env.models import DroneState

def test_grader():
    # Test perfect state
    state_perfect = DroneState(
        deliveries_total=1,
        deliveries_done=1,
        battery=1.0,
        step_count=0
    )
    score_p = compute_grade(state_perfect, 60.0)
    print(f"Perfect score: {score_p}")
    assert 0 < score_p < 1

    # Test failure state
    state_fail = DroneState(
        deliveries_total=1,
        deliveries_done=0,
        battery=0.0,
        step_count=60
    )
    score_f = compute_grade(state_fail, 60.0)
    print(f"Failure score: {score_f}")
    assert 0 < score_f < 1

    # Test intermediate
    state_mid = DroneState(
        deliveries_total=1,
        deliveries_done=0,
        battery=0.5,
        step_count=30
    )
    score_m = compute_grade(state_mid, 60.0)
    print(f"Mid score: {score_m}")
    assert 0 < score_m < 1

    print("All grader tests passed!")

if __name__ == "__main__":
    test_grader()
