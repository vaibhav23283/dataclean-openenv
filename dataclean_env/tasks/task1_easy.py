import pandas as pd

def get_task():
    data = {
        "name":   ["Alice", "Bob",  "Alice", "Charlie", "Bob"],
        "age":    [25,      None,   25,      30,         None],
        "salary": [50000,   60000,  50000,   70000,      60000]
    }
    return {
        "name":          "Employee Data - Remove Duplicates and Fill Nulls",
        "dirty_data":    pd.DataFrame(data),
        "expected_dups": 2,
        "description":   "Remove 2 duplicate rows, fill 2 missing ages with mean"
    }

def grade(df: pd.DataFrame) -> float:
    score = 0.0
    if df.duplicated().sum() == 0:
        score += 0.5
    if df["age"].isnull().sum() == 0:
        score += 0.5
    return round(score, 2)