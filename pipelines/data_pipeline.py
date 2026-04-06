"""
Shared Data Pipeline — ClearSight Analytics
=============================================
Data loading, cleaning, feature engineering, and splitting for Models 1 & 2.
Reused in both train.py and predict.py to ensure consistent preprocessing.

Usage:
    from pipelines.data_pipeline import load_and_clean, engineer_features, split_data
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# =============================================================================
# STEP 1: Load and Clean
# =============================================================================

def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load patient encounters CSV and apply all cleaning steps.
    Works on both training data and unseen test data.
    """
    df = pd.read_csv(filepath)

    # 1. Replace '?' with NaN (hidden missing values)
    df.replace('?', np.nan, inplace=True)

    # 2. Exclude death/hospice patients (cannot be readmitted = data leakage)
    exclude_dispositions = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(exclude_dispositions)]

    # 3. Create binary readmission target
    #    '<30' and '>30' -> 1 (readmitted), 'NO' -> 0 (not readmitted)
    if 'readmitted' in df.columns:
        df['readmission_binary'] = (df['readmitted'] != 'NO').astype(int)

    # 4. Convert age brackets to numeric midpoints
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25,
        '[30-40)': 35, '[40-50)': 45, '[50-60)': 55,
        '[60-70)': 65, '[70-80)': 75, '[80-90)': 85,
        '[90-100)': 95
    }
    df['age_numeric'] = df['age'].map(age_map)

    # 5. Drop columns we won't use
    drop_cols = [
        'encounter_id',   # just an ID, not a feature
        'patient_nbr',    # patient ID, not a feature
        'weight',         # 97% missing
        'payer_code',     # 40% missing, not clinically useful
        'readmitted',     # replaced by readmission_binary
        'age',            # replaced by age_numeric
        'examide',        # zero variance
        'citoglipton',    # zero variance
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    print(f"Loaded and cleaned: {len(df)} rows, {len(df.columns)} columns")
    return df

# =============================================================================
# STEP 2: Feature Engineering
# =============================================================================

def categorize_icd9(code) -> str:
    """Group ICD-9 diagnostic codes into clinical categories."""
    if pd.isna(code):
        return 'missing'
    code = str(code).strip()

    if code.startswith('250'):
        return 'diabetes'
    try:
        numeric = float(code[:3])
    except ValueError:
        if code.startswith('V') or code.startswith('E'):
            return 'external'
        return 'other'

    if 390 <= numeric <= 459:
        return 'circulatory'
    elif 460 <= numeric <= 519:
        return 'respiratory'
    elif 520 <= numeric <= 579:
        return 'digestive'
    elif 580 <= numeric <= 629:
        return 'genitourinary'
    elif 710 <= numeric <= 739:
        return 'musculoskeletal'
    elif 800 <= numeric <= 999:
        return 'injury'
    else:
        return 'other'


def engineer_features(df: pd.DataFrame, encoders=None):
        
    """
    Create all engineered features for Models 1 & 2.
    Must work on both training data and unseen test data.
    """
    # --- Medication features ---
    
    med_cols = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
                'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
                'miglitol', 'troglitazone', 'tolazamide', 'insulin',
                'glyburide-metformin', 'glipizide-metformin',
                'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                'metformin-pioglitazone']
    med_cols = [c for c in med_cols if c in df.columns]

    df['n_meds_changed'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v != 'No'), axis=1
    )
    df['any_med_changed'] = (df['n_meds_changed'] > 0).astype(int)
    df['n_meds_increased'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v == 'Up'), axis=1
    )
    df['n_meds_decreased'] = df[med_cols].apply(
        lambda row: sum(1 for v in row if v == 'Down'), axis=1
    )

    # Drop individual medication columns (replaced by aggregates)
    df.drop(columns=med_cols, inplace=True)

    # --- Clinical test missingness flags ---
    df['glucose_tested'] = (df['max_glu_serum'] != 'None').astype(int) if 'max_glu_serum' in df.columns else 0
    df['a1c_tested'] = (df['A1Cresult'] != 'None').astype(int) if 'A1Cresult' in df.columns else 0

    # Encode test results as ordinal
    glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    if 'max_glu_serum' in df.columns:
        df['max_glu_serum'] = df['max_glu_serum'].map(glu_map).fillna(0).astype(int)
    if 'A1Cresult' in df.columns:
        df['A1Cresult'] = df['A1Cresult'].map(a1c_map).fillna(0).astype(int)

    # --- ICD-9 diagnosis categories ---
    for diag_col in ['diag_1', 'diag_2', 'diag_3']:
        if diag_col in df.columns:
            df[f'{diag_col}_cat'] = df[diag_col].apply(categorize_icd9)
            df.drop(columns=[diag_col], inplace=True)

    # --- Clinical complexity score ---
    df['complexity_score'] = (
        df['num_lab_procedures'].fillna(0) / 50 +
        df['num_procedures'].fillna(0) / 6 +
        df['num_medications'].fillna(0) / 20 +
        df['number_diagnoses'].fillna(0) / 9 +
        df['number_emergency'].fillna(0) / 3 +
        df['number_inpatient'].fillna(0) / 5
    )

    # --- Encode change and diabetesMed as binary ---
    if 'change' in df.columns:
        df['change'] = (df['change'] == 'Ch').astype(int)
    if 'diabetesMed' in df.columns:
        df['diabetesMed'] = (df['diabetesMed'] == 'Yes').astype(int)

    # --- Encode remaining categorical columns ---
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ['readmitted']]

    if encoders is None:
        # TRAINING MODE: fit new encoders and return them
        encoders = {}
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        # PREDICTION MODE: use existing encoders from training
        for col in cat_cols:
            df[col] = df[col].fillna('Unknown')
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            else:
                df[col] = 0

    # --- Fill any remaining NaN with median per column ---
    for col in df.select_dtypes(include='number').columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    print(f"Feature engineering complete: {len(df.columns)} columns")
    return df, encoders

# =============================================================================
# STEP 3: Split Data
# =============================================================================

def split_data(df: pd.DataFrame, target_col: str = 'readmission_binary',
               test_size: float = 0.2, random_state: int = 42):
    """
    Split into X_train, X_val, y_train, y_val with stratification.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train: {len(X_train)} rows | Val: {len(X_val)} rows")
    print(f"Train target distribution:\n{y_train.value_counts(normalize=True).round(3)}")
    return X_train, X_val, y_train, y_val


# =============================================================================
# STEP 4: Full Pipeline (convenience function)
# =============================================================================

def prepare_training_data(filepath: str):
    """
    Run the full pipeline: load → clean → engineer → split.
    Used by train.py for both Model 1 and Model 2.

    Returns: X_train, X_val, y_train, y_val, encoders
    """
    df = load_and_clean(filepath)
    df, encoders = engineer_features(df)
    X_train, X_val, y_train, y_val = split_data(df)
    return X_train, X_val, y_train, y_val, encoders


def prepare_test_data(filepath: str, encoders: dict):
    """
    Run the pipeline WITHOUT splitting (for predict.py on unseen data).
    Uses encoders fitted during training for consistent encoding.

    Returns the full feature matrix ready for prediction.
    """
    df = load_and_clean(filepath)
    df, _ = engineer_features(df, encoders=encoders)

    if 'readmission_binary' in df.columns:
        df.drop(columns=['readmission_binary'], inplace=True)

    print(f"Test data prepared: {len(df)} rows, {len(df.columns)} columns")
    return df

