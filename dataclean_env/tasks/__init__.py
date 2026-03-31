from dataclean_env.tasks.task1_easy import get_task as task1, grade as grade1
from dataclean_env.tasks.task2_medium import get_task as task2, grade as grade2
from dataclean_env.tasks.task3_hard import get_task as task3, grade as grade3

TASKS = {
    "task1_easy":   {**task1(), "grade": grade1},
    "task2_medium": {**task2(), "grade": grade2},
    "task3_hard":   {**task3(), "grade": grade3},
}