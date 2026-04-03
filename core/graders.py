"""
drone_delivery_env/core/graders.py
Evaluation logic for scoring Drone Delivery missions.
"""
from drone_env.models import DroneState


def compute_grade(state: DroneState, max_steps: float) -> float:
    """
    Unified grade calculation:
    - 70% weighted by deliveries completed.
    - 30% weighted by efficiency (remaining battery and steps).
    """
    if state.deliveries_total == 0: return 1.0
    
    delivery_ratio = state.deliveries_done / state.deliveries_total
    
    # Efficiency factor (0.0 to 1.0)
    # Penalize if steps taken > average or battery is very low
    efficiency = state.battery * 0.5 + (1.0 - (state.step_count / max_steps)) * 0.5
    efficiency = max(0.0, min(1.0, efficiency))
    
    score = (delivery_ratio * 0.8) + (efficiency * 0.2)
    
    # If not all deliveries are done, the maximum score is capped
    if state.deliveries_done < state.deliveries_total:
        score = min(score, 0.49)
        
    return max(0.0, min(1.0, score))


def grade_easy(state: DroneState) -> float:
    return compute_grade(state, 60.0)


def grade_medium(state: DroneState) -> float:
    return compute_grade(state, 100.0)


def grade_hard(state: DroneState) -> float:
    return compute_grade(state, 160.0)


GRADERS = {
    "easy_delivery": grade_easy,
    "medium_delivery": grade_medium,
    "hard_delivery": grade_hard
}
