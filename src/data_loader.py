
import pandas as pd
from pathlib import Path


def load_data(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    if file_path.suffix.lower() == ".txt":
        df = pd.read_csv(
            file_path,
            sep="|",
            quotechar='"'
        )

    elif file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)

    else:
        raise ValueError(
            "Only TXT and CSV files are supported."
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    print(f"Loaded {len(df)} records successfully.")

    return df

def identify_columns(df, column_mapping):
    """Identify dataset columns using candidate names."""

    identified = {}

    for standard_name, candidates in column_mapping.items():

        for candidate in candidates:
            candidate = candidate.lower().replace(" ", "_")

            if candidate in df.columns:
                identified[standard_name] = candidate
                break

    return identified


def validate_data(df, columns):
    """Check whether essential data is available."""

    if "text" not in columns:
        raise ValueError(
            "No message column found. Check COLUMN_MAPPING."
        )

    text_column = columns["text"]

    if df[text_column].dropna().empty:
        raise ValueError("No valid social media messages found.")

    return True


def inspect_data(df):
    """Display dataset information."""

    print("\n========== DATASET INFORMATION ==========")

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nFirst five records:")
    print(df.head())