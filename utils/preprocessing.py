import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the input DataFrame by stripping whitespace from string columns
    and performing basic preprocessing.
    """

    # Automatically select all columns of object type (string columns)
    string_columns = df.select_dtypes(include=['object']).columns.tolist()

    for col in string_columns:
        # Strip whitespace safely, ignoring NaNs
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x)

    # Optional: drop rows with missing target column
    if 'rating' in df.columns:
        df = df.dropna(subset=['rating'])

    return df
