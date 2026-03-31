import pandas as pd

def get_task():
    data = {
        "name":       ["Alice", "Bob", "Alice", "Charlie", "Dave", "Eve"],
        "age":        [25,      None,  25,      30,        35,     None],
        "salary":     [50000,   60000, 50000,   70000,     999999, 65000],
        "department": ["HR",    None,  "HR",    "IT",      "IT",   None]
    }
    return {
        "name":        "Full Pipeline - Duplicates, Nulls and Outliers",
        "dirty_data":  pd.DataFrame(data),
        "description": "Remove duplicates, fill nulls, drop salary outliers"
    }

def grade(df: pd.DataFrame) -> float:
    score = 0.0
    if df.duplicated().sum() == 0:
        score += 0.34
    if df["age"].isnull().sum() == 0:
        score += 0.33
    if df["salary"].max() < 200000:
        score += 0.33
    return round(score, 2)