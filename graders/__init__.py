from .easy import grade_easy
from .medium import grade_medium
from .hard import grade_hard

GRADERS = {
    "drone_env.graders.easy:grade_easy": grade_easy,
    "drone_env.graders.medium:grade_medium": grade_medium,
    "drone_env.graders.hard:grade_hard": grade_hard,
}

__all__ = ["grade_easy", "grade_medium", "grade_hard", "GRADERS"]
