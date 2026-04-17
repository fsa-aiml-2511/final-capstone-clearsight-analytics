"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, preprocess, split_data
"""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(filename):
    """Load a raw CSV file from data/raw/.

    Args:
        filename: Name of the CSV file (e.g., "patient_medication_feedback.csv")

    Returns:
        pandas DataFrame
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you've downloaded the data to data/raw/"
        )
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows from {filename}")
    return df


def clean_data(df):
    """Clean the patient medication feedback dataset.

    - Drops rows missing the review text or target label
    - Removes rows where the review text is empty/whitespace
    - Resets the index

    Returns:
        Cleaned DataFrame
    """
    target_col = "effectiveness_3class"
    text_col = "benefitsReview"

    df = df.dropna(subset=[text_col, target_col])
    df = df[df[text_col].str.strip().str.len() > 0]
    df = df.reset_index(drop=True)

    print(f"After cleaning: {len(df)} rows")
    print(df[target_col].value_counts())
    return df


def engineer_features(df):
    """Add derived features useful for analysis alongside the NLP model.

    - review_word_count: number of words in the review
    - review_char_count: number of characters in the review

    Returns:
        DataFrame with added feature columns
    """
    df["review_word_count"] = df["benefitsReview"].str.split().str.len()
    df["review_char_count"] = df["benefitsReview"].str.len()
    return df


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and validation sets.

    IMPORTANT: Use stratify=y for imbalanced classification tasks.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion for validation (default 0.2)
        random_state: For reproducibility

    Returns:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_processed_data(df, filename):
    """Save processed data to data/processed/.

    Args:
        df: Processed DataFrame
        filename: Output filename (e.g., "encounters_processed.csv")
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filename):
    """Load previously processed data from data/processed/.

    Args:
        filename: Name of the processed CSV file

    Returns:
        pandas DataFrame
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)
