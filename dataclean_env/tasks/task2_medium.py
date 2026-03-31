import pandas as pd

def get_task():
    data = {
        "product": ["Apple", "Banana", "Cherry", "Date", "Elder"],
        "price":   ["$10.5", "$20.0", "$15.75", "$8.0", "$30.0"],
        "date":    ["01/15/2024", "2024-02-20", "March 3 2024",
                    "2024-04-10", "05/05/2024"]
    }
    return {
        "name":        "Product Data - Fix Data Types and Dates",
        "dirty_data":  pd.DataFrame(data),
        "description": "Fix price dtype from string to float, standardize dates"
    }

def grade(df: pd.DataFrame) -> float:
    score = 0.0
    try:
        df["price"].astype(float)
        score += 0.5
    except:
        pass
    try:
        pd.to_datetime(df["date"])
        score += 0.5
    except:
        pass
    return round(score, 2)