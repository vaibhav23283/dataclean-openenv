from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class Observation(BaseModel):
    """What the agent SEES at each step."""
    task_id: str                          # which task: "task1_easy" etc.
    dataset_name: str                     # human readable name
    current_data: List[Dict[str, Any]]   # CSV rows as list of dicts
    issues_detected: List[str]            # e.g. ["3 null values in 'age'"]
    step_number: int                      # steps taken so far
    done: bool                            # True when episode is over


class Action(BaseModel):
    """What the agent CAN DO — sent to /step endpoint."""
    operation: str
    # Valid values:
    # "drop_duplicates"  — no parameters needed
    # "fill_null"        — needs: {"column": "age", "method": "mean"}
    # "fix_dtype"        — needs: {"column": "price", "dtype": "float"}
    # "drop_outliers"    — needs: {"column": "salary"}
    # "done"             — agent signals it finished
    parameters: Optional[Dict[str, Any]] = {}


class Reward(BaseModel):
    """Score the agent gets after each action."""
    score: float           # 0.0 to 1.0
    partial_credit: float  # reward for partial progress
    message: str           # e.g. "Removed 2 duplicate rows"