# DataClean-OpenEnv 🧹

Real-world data cleaning environment for AI agents.
Built for Scaler × Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026.

## Tasks

| Task ID | Difficulty | Objective |
|---------|-----------|-----------|
| task1_easy | Easy | Remove duplicates + fill null ages |
| task2_medium | Medium | Fix price dtype + standardize dates |
| task3_hard | Hard | Dups + nulls + salary outliers |

## Action Space

| Operation | Parameters |
|-----------|-----------|
| drop_duplicates | none |
| fill_null | column, method (mean/mode/drop) |
| fix_dtype | column, dtype (float/datetime) |
| drop_outliers | column |
| done | none |

## Run Locally

```bash
pip install -r requirements.txt
uvicorn dataclean_env.env:app --host 0.0.0.0 --port 8000
```

## Run Inference

```bash
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export ENV_URL=http://localhost:8000
python inference.py
```

## Baseline Scores

| Task | Score |
|------|-------|
| task1_easy | ~0.80 |
| task2_medium | ~0.65 |
| task3_hard | ~0.55 |