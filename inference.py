"""
inference.py — DataClean OpenEnv
RULES:
  - Named exactly: inference.py
  - In ROOT folder (same level as Dockerfile)
  - Must finish all 3 tasks in under 20 minutes
"""
import os
import json
import time
import requests

# ── Read credentials from environment variables ──────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "hf_dummy_key_placeholder"
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

from openai import OpenAI
try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception as e:
    print(f"Warning: {e}")
    client = None

TASKS     = ["task1_easy", "task2_medium", "task3_hard"]
MAX_STEPS = 8

SYSTEM_PROMPT = """
You are a data cleaning expert agent.
Given dataset issues, pick the single best cleaning operation.
Reply ONLY with valid JSON. No explanation. No extra text.

Valid responses:
{"operation": "drop_duplicates", "parameters": {}}
{"operation": "fill_null", "parameters": {"column": "age", "method": "mean"}}
{"operation": "fill_null", "parameters": {"column": "department", "method": "mode"}}
{"operation": "fix_dtype", "parameters": {"column": "price", "dtype": "float"}}
{"operation": "fix_dtype", "parameters": {"column": "date", "dtype": "datetime"}}
{"operation": "drop_outliers", "parameters": {"column": "salary"}}
{"operation": "done", "parameters": {}}

Rules:
- "duplicate rows" found          → drop_duplicates
- null in numeric column (age)    → fill_null, method=mean
- null in text column (dept)      → fill_null, method=mode
- price has $ signs               → fix_dtype, dtype=float
- inconsistent date formats       → fix_dtype, dtype=datetime
- extreme salary values           → drop_outliers
- no issues left                  → done
"""

def ask_llm(issues: list, data: list) -> dict:
    """Ask LLM which cleaning action to take. Falls back to rule-based if LLM fails."""
    # ── Try LLM first ────────────────────────────────────────────────────────
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model    = MODEL_NAME,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content":
                        f"Issues: {json.dumps(issues)}\n"
                        f"First 3 rows: {json.dumps(data[:3])}\n"
                        f"What is the best next cleaning operation?"}
                ],
                max_tokens=100, temperature=0.1
            )
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"    LLM call failed: {e} — using rule-based fallback")

    # ── Rule-based fallback (works even without LLM) ─────────────────────────
    return rule_based_action(issues, data)


def rule_based_action(issues: list, data: list) -> dict:
    """
    Simple rule-based agent as fallback.
    Reads the issues list and picks the most appropriate action.
    This ensures inference.py always produces scores even if LLM is unavailable.
    """
    if not issues:
        return {"operation": "done", "parameters": {}}

    for issue in issues:
        issue_lower = issue.lower()

        # Rule 1: duplicates
        if "duplicate" in issue_lower:
            return {"operation": "drop_duplicates", "parameters": {}}

        # Rule 2: null values — figure out which column and type
        if "null values in" in issue_lower:
            # Extract column name from string like "3 null values in 'age'"
            try:
                col = issue.split("'")[1]
            except IndexError:
                col = None

            if col:
                # Numeric columns → mean, text columns → mode
                numeric_cols = ["age", "salary", "price", "score", "amount"]
                method = "mean" if any(nc in col.lower() for nc in numeric_cols) else "mode"
                return {"operation": "fill_null",
                        "parameters": {"column": col, "method": method}}

        # Rule 3: price/amount dtype issues (detected from data)
        if data and col_has_dollar(data, "price"):
            return {"operation": "fix_dtype",
                    "parameters": {"column": "price", "dtype": "float"}}

        # Rule 4: date standardization
        if data and col_has_mixed_dates(data, "date"):
            return {"operation": "fix_dtype",
                    "parameters": {"column": "date", "dtype": "datetime"}}

        # Rule 5: salary outliers
        if data and col_has_outliers(data, "salary"):
            return {"operation": "drop_outliers",
                    "parameters": {"column": "salary"}}

    return {"operation": "done", "parameters": {}}


def col_has_dollar(data: list, col: str) -> bool:
    """Check if a column contains $ signs."""
    try:
        return any("$" in str(row.get(col, "")) for row in data)
    except Exception:
        return False


def col_has_mixed_dates(data: list, col: str) -> bool:
    """Check if date column has inconsistent formats."""
    try:
        dates = [str(row.get(col, "")) for row in data if row.get(col)]
        formats = set()
        for d in dates:
            if "/" in d:
                formats.add("slash")
            elif "-" in d:
                formats.add("dash")
            elif any(m in d for m in ["Jan","Feb","Mar","Apr","May","Jun",
                                       "Jul","Aug","Sep","Oct","Nov","Dec"]):
                formats.add("written")
        return len(formats) > 1
    except Exception:
        return False


def col_has_outliers(data: list, col: str) -> bool:
    """Check if a numeric column has extreme values."""
    try:
        vals = [float(row[col]) for row in data if row.get(col) is not None]
        if not vals:
            return False
        return max(vals) > 500000 or min(vals) < 0
    except Exception:
        return False


def run_task(task_id: str) -> float:
    """Run one full episode. Returns average score."""
    print(f"\n{'='*55}")
    print(f"  Task: {task_id}")
    print(f"{'='*55}")

    # Reset environment
    try:
        obs = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=30
        ).json()
    except Exception as e:
        print(f"  ERROR: Could not connect to environment at {ENV_URL}")
        print(f"  Details: {e}")
        return 0.0

    total, steps = 0.0, 0

    for i in range(MAX_STEPS):
        issues = obs.get("issues_detected", [])
        data   = obs.get("current_data", [])
        done   = obs.get("done", False)

        print(f"\n  Step {i+1} — Issues: {issues}")

        if not issues or done:
            print("  No issues remaining — task complete.")
            break

        action = ask_llm(issues, data)
        print(f"  Action chosen: {json.dumps(action)}")

        if action.get("operation") == "done":
            print("  Agent signalled done.")
            break

        try:
            result = requests.post(
                f"{ENV_URL}/step",
                params={"task_id": task_id},
                json=action,
                timeout=30
            ).json()
        except Exception as e:
            print(f"  ERROR: Step request failed: {e}")
            break

        obs    = result["observation"]
        score  = result["reward"]["score"]
        msg    = result["reward"]["message"]
        total += score
        steps += 1

        print(f"  Score: {score:.2f} | {msg}")

        if result["done"]:
            print("  Episode ended.")
            break

        time.sleep(0.5)

    avg = total / max(steps, 1)
    print(f"\n  Result: {avg:.2f} avg over {steps} steps")
    return avg


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  DataClean OpenEnv — Inference Script")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Env URL: {ENV_URL}")
    print(f"{'='*55}")

    results = {}
    for task in TASKS:
        results[task] = run_task(task)

    print(f"\n{'='*55}")
    print(f"  FINAL SCORES")
    print(f"{'='*55}")
    for task, score in results.items():
        bar = "█" * int(score * 25)
        print(f"  {task:<22}  {score:.2f}  {bar}")
    overall = sum(results.values()) / len(results)
    print(f"\n  Overall Average: {overall:.2f}")
    print(f"{'='*55}\n")
