def grade_easy(state):
    from drone_env.graders.easy import grade_easy as fn
    return fn(state)

def grade_medium(state):
    from drone_env.graders.medium import grade_medium as fn
    return fn(state)

def grade_hard(state):
    from drone_env.graders.hard import grade_hard as fn
    return fn(state)

__all__ = ["grade_easy", "grade_medium", "grade_hard"]
