import copy
import math
from fastapi import FastAPI
from dataclean_env.models import Observation, Action, Reward
from dataclean_env.tasks import TASKS
import pandas as pd

app = FastAPI(title="DataClean OpenEnv", version="1.0.0")


class DataCleanEnv:

    def __init__(self, task_id: str):
        self.task_id    = task_id
        self.task       = TASKS[task_id]
        self.df         = copy.deepcopy(self.task["dirty_data"])
        self.step_count = 0
        self.max_steps  = 10
        self.done       = False

    def reset(self) -> Observation:
        self.df         = copy.deepcopy(self.task["dirty_data"])
        self.step_count = 0
        self.done       = False
        return self._build_obs()

    def step(self, action: Action):
        reward = self._apply(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        return self._build_obs(), reward, self.done, {}

    def state(self) -> dict:
        return {
            "task_id": self.task_id,
            "step":    self.step_count,
            "done":    self.done,
            "rows":    len(self.df),
            "columns": list(self.df.columns),
        }

    def _build_obs(self) -> Observation:
        clean_data = []
        for row in self.df.to_dict(orient="records"):
            clean_row = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row.items()
            }
            clean_data.append(clean_row)
        return Observation(
            task_id         = self.task_id,
            dataset_name    = self.task["name"],
            current_data    = clean_data,
            issues_detected = self._issues(),
            step_number     = self.step_count,
            done            = self.done
        )

    def _issues(self) -> list:
        issues = []
        dups = self.df.duplicated().sum()
        if dups > 0:
            issues.append(f"{dups} duplicate rows found")
        for col in self.df.columns:
            n = self.df[col].isnull().sum()
            if n > 0:
                issues.append(f"{n} null values in '{col}'")
        return issues

    def _apply(self, action: Action) -> Reward:
        op = action.operation
        p  = action.parameters or {}

        if op == "drop_duplicates":
            before  = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = before - len(self.df)
            if removed > 0:
                return Reward(score=0.8, partial_credit=0.4,
                              message=f"Removed {removed} duplicate rows")
            return Reward(score=0.1, partial_credit=0.05,
                          message="No duplicates found")

        elif op == "fill_null":
            col    = p.get("column")
            method = p.get("method", "mean")
            if not col or col not in self.df.columns:
                return Reward(score=0.0, partial_credit=0.0,
                              message=f"Column '{col}' not found")
            count = int(self.df[col].isnull().sum())
            if method == "mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif method == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif method == "drop":
                self.df = self.df.dropna(subset=[col])
            return Reward(score=0.7, partial_credit=0.35,
                          message=f"Filled {count} nulls in '{col}' using {method}")

        elif op == "fix_dtype":
            col   = p.get("column")
            dtype = p.get("dtype", "float")
            if not col or col not in self.df.columns:
                return Reward(score=0.0, partial_credit=0.0,
                              message=f"Column '{col}' not found")
            try:
                if dtype == "float":
                    self.df[col] = (self.df[col]
                                    .astype(str)
                                    .str.replace("$", "", regex=False)
                                    .str.strip()
                                    .astype(float))
                elif dtype == "datetime":
                    self.df[col] = (pd.to_datetime(self.df[col],
                                                   infer_datetime_format=True)
                                      .dt.strftime("%Y-%m-%d"))
                else:
                    self.df[col] = self.df[col].astype(dtype)
                return Reward(score=0.75, partial_credit=0.375,
                              message=f"Converted '{col}' to {dtype}")
            except Exception as e:
                return Reward(score=0.0, partial_credit=0.0,
                              message=f"Conversion failed: {str(e)}")

        elif op == "drop_outliers":
            col = p.get("column")
            if not col or col not in self.df.columns:
                return Reward(score=0.0, partial_credit=0.0,
                              message=f"Column '{col}' not found")
            Q1  = self.df[col].quantile(0.25)
            Q3  = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            before  = len(self.df)
            self.df = self.df[
                (self.df[col] >= Q1 - 1.5 * IQR) &
                (self.df[col] <= Q3 + 1.5 * IQR)
            ]
            removed = before - len(self.df)
            score   = 0.9 if removed > 0 else 0.1
            return Reward(score=score, partial_credit=score * 0.5,
                          message=f"Removed {removed} outliers from '{col}'")

        elif op == "done":
            self.done = True
            return Reward(score=0.5, partial_credit=0.25,
                          message="Agent signalled task complete")

        return Reward(score=0.0, partial_credit=0.0,
                      message=f"Unknown operation: '{op}'")


_envs: dict = {}


@app.get("/")
def health():
    return {"status": "ok", "env": "dataclean-openenv",
            "tasks": list(TASKS.keys())}

@app.post("/reset")
def reset(task_id: str = "task1_easy"):
    if task_id not in TASKS:
        return {"error": f"Unknown task. Choose from: {list(TASKS.keys())}"}
    _envs[task_id] = DataCleanEnv(task_id)
    return _envs[task_id].reset().dict()

@app.post("/step")
def step(task_id: str, action: Action):
    if task_id not in _envs:
        _envs[task_id] = DataCleanEnv(task_id)
        _envs[task_id].reset()
    obs, reward, done, info = _envs[task_id].step(action)
    return {"observation": obs.dict(), "reward": reward.dict(),
            "done": done, "info": info}

@app.get("/state")
def state(task_id: str = "task1_easy"):
    if task_id not in _envs:
        return {"error": "Call /reset first"}
    return _envs[task_id].state()
@app.get("/schema")
def schema():
    """Return openenv schema — required by validator."""
    return {
        "name": "dataclean-openenv",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "endpoints": {
            "reset": "/reset",
            "step": "/step",
            "state": "/state"
        },
        "actions": [
            "drop_duplicates",
            "fill_null",
            "fix_dtype",
            "drop_outliers",
            "done"
        ]
    }

@app.get("/openenv.yaml")
def openenv_yaml():
    """Serve openenv.yaml — required by validator."""
    import os
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "openenv.yaml")
    with open(yaml_path, "r") as f:
        content = f.read()
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content)