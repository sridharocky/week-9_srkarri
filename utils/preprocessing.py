import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning of the raw coffee review data:
    - Drop rows where all values are NaN
    - Normalize column names
    """
    df = df.dropna(how="all")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_")
    return df
