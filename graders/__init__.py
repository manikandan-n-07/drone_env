from .easy import grade_easy
from .medium import grade_medium
from .hard import grade_hard

GRADERS = {
    "graders:grade_easy": grade_easy,
    "graders:grade_medium": grade_medium,
    "graders:grade_hard": grade_hard,
}

__all__ = ["grade_easy", "grade_medium", "grade_hard", "GRADERS"]
