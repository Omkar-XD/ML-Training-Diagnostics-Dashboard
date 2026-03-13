import pandas as pd
from sklearn.model_selection import train_test_split

def basic_eda(df: pd.DataFrame):
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isna().sum().to_dict(),
        "head": df.head().to_dict(orient="records"),
    }

def handle_missing(df: pd.DataFrame, strategy: str = "drop"):
    if strategy == "drop":
        return df.dropna()
    if strategy == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if strategy == "median":
        return df.fillna(df.median(numeric_only=True))
    if strategy == "mode":
        return df.fillna(df.mode().iloc[0])
    return df

def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed)
