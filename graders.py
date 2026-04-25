"""
Unified graders for Drone Delivery OpenEnv.
All grading functions are consolidated here for reliable importing during validation.
"""

def _get_attr(state, key, default=0):
    """Safely get an attribute from a state object or dict."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def compute_grade(state, max_steps: float) -> float:
    """
    Unified grade calculation:
    - 80% weighted by deliveries completed.
    - 20% weighted by efficiency (remaining battery and steps).
    """
    deliveries_total = _get_attr(state, "deliveries_total", 0)
    deliveries_done  = _get_attr(state, "deliveries_done",  0)
    drones           = _get_attr(state, "drones",          [])
    step_count       = _get_attr(state, "step_count",       0)

    # Average battery of all drones
    if drones:
        battery = sum(d.battery for d in drones) / len(drones)
    else:
        battery = 1.0

    if deliveries_total == 0:
        return 0.5

    delivery_ratio = deliveries_done / deliveries_total
    efficiency = float(battery) * 0.5 + (1.0 - (float(step_count) / float(max_steps))) * 0.5
    efficiency = max(0.0, min(1.0, efficiency))

    score = (delivery_ratio * 0.8) + (efficiency * 0.2)
    if deliveries_done < deliveries_total:
        score = min(score, 0.49)

    return max(0.01, min(0.99, float(score)))


def grade_easy(state) -> float:
    """Grader for easy_delivery task (12x12 grid, 3 deliveries, 100 max steps)."""
    return compute_grade(state, 100.0)


def grade_medium(state) -> float:
    """Grader for medium_delivery task (16x16 grid, 5 deliveries, 150 max steps)."""
    return compute_grade(state, 150.0)


def grade_hard(state) -> float:
    """Grader for hard_delivery task (20x20 grid, 10 deliveries, 250 max steps)."""
    return compute_grade(state, 250.0)

# --- Standardization Mapping ---
GRADERS = {
    "graders:grade_easy": grade_easy,
    "graders:grade_medium": grade_medium,
    "graders:grade_hard": grade_hard,
}
