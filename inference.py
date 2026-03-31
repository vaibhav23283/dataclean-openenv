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
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")

client    = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
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
        raw = raw.replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except Exception as e:
        print(f"    LLM error: {e}")
        return {"operation": "done", "parameters": {}}

def run_task(task_id: str) -> float:
    print(f"\n{'='*55}\n  Task: {task_id}\n{'='*55}")
    try:
        obs = requests.post(f"{ENV_URL}/reset",
                            params={"task_id": task_id}, timeout=30).json()
    except Exception as e:
        print(f"  Connection failed: {e}")
        return 0.0
    total, steps = 0.0, 0
    for i in range(MAX_STEPS):
        issues = obs.get("issues_detected", [])
        print(f"\n  Step {i+1}: {issues}")
        if not issues or obs.get("done"):
            break
        action = ask_llm(issues, obs.get("current_data", []))
        print(f"  Action: {action}")
        if action.get("operation") == "done":
            break
        try:
            result = requests.post(f"{ENV_URL}/step",
                                   params={"task_id": task_id},
                                   json=action, timeout=30).json()
        except Exception as e:
            print(f"  Step failed: {e}")
            break
        obs    = result["observation"]
        score  = result["reward"]["score"]
        total += score
        steps += 1
        print(f"  Score: {score:.2f} | {result['reward']['message']}")
        if result["done"]:
            break
        time.sleep(0.5)
    avg = total / max(steps, 1)
    print(f"\n  Avg: {avg:.2f} over {steps} steps")
    return avg

if __name__ == "__main__":
    print(f"\nDataClean OpenEnv — Inference\nModel: {MODEL_NAME}\nEnv: {ENV_URL}")
    results = {t: run_task(t) for t in TASKS}
    print(f"\n{'='*55}\n  FINAL SCORES\n{'='*55}")
    for t, s in results.items():
        print(f"  {t:<22} {s:.2f}  {'█'*int(s*25)}")
    print(f"\n  Overall: {sum(results.values())/len(results):.2f}\n{'='*55}")