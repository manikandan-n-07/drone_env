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
    battery          = _get_attr(state, "battery",          1.0)
    step_count       = _get_attr(state, "step_count",       0)

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
    """Grader for easy_delivery task (10x10 grid, 1 delivery, 60 max steps)."""
    return compute_grade(state, 60.0)


def grade_medium(state) -> float:
    """Grader for medium_delivery task (14x14 grid, 3 deliveries, 100 max steps)."""
    return compute_grade(state, 100.0)


def grade_hard(state) -> float:
    """Grader for hard_delivery task (18x18 grid, 5 deliveries, 160 max steps)."""
    return compute_grade(state, 160.0)

# --- Standardization Mapping ---
GRADERS = {
    "graders:grade_easy": grade_easy,
    "graders:grade_medium": grade_medium,
    "graders:grade_hard": grade_hard,
}
