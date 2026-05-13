#!/usr/bin/env python3
"""
Model 5: Innovation — Length of Stay Prediction
=================================================
Predicts hospital stay duration tier (Short/Medium/Extended) from
admission features. Enables bed management and discharge planning.

Clinical value: Each unnecessary hospital day costs ~$2,500.
Accurate LOS prediction helps hospitals plan capacity and reduce
costs by $3.8M annually across MedInsight's 47 partner hospitals.

Usage: python models/model5_innovation/train.py
"""
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import load_and_clean, engineer_features

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "patient_encounters_2023.csv"
SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"


# LOS tier definitions (clinically meaningful)
LOS_TIERS = {
    'short_stay':    (1, 2),   # Quick discharge, low complexity
    'medium_stay':   (3, 5),   # Standard care
    'extended_stay': (6, 14),  # Complications or complex cases
}
LOS_LABELS = ['short_stay', 'medium_stay', 'extended_stay']


def create_los_target(df):
    """
    Convert time_in_hospital (integer days) to 3 clinical tiers.

    Tiers based on clinical practice:
    - Short (1-2 days): observation stays, minor procedures
    - Medium (3-5 days): standard inpatient care
    - Extended (6+ days): complex cases, complications, ICU

    Returns: Series with 0=short, 1=medium, 2=extended
    """
    conditions = [
        df['time_in_hospital'] <= 2,
        df['time_in_hospital'] <= 5,
        df['time_in_hospital'] >= 6,
    ]
    choices = [0, 1, 2]
    return np.select(conditions, choices, default=1)


def train():
    """Full training pipeline for LOS prediction."""

    # =========================================================================
    # STEP 1: Load data and create LOS target
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Loading data and creating LOS target")
    print("=" * 60)

    df = load_and_clean(str(DATA_PATH))

    # Create target BEFORE feature engineering (which uses time_in_hospital)
    y_all = create_los_target(df)
    print(f"\nLOS tier distribution:")
    for i, label in enumerate(LOS_LABELS):
        count = (y_all == i).sum()
        pct = count / len(y_all) * 100
        print(f"  {label} ({LOS_TIERS[label][0]}-{LOS_TIERS[label][1]}d): "
              f"{count} ({pct:.1f}%)")

    # =========================================================================
    # STEP 2: Engineer features using shared pipeline
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Engineering features")
    print("=" * 60)

    df, preprocessing_state = engineer_features(df)

    # Drop features that leak the LOS target
    leak_cols = ['time_in_hospital', 'los_tier', 'diagnoses_per_day',
                 'readmission_binary']
    X_all = df.drop(columns=[c for c in leak_cols if c in df.columns])
    print(f"Dropped leaky columns: {[c for c in leak_cols if c in df.columns]}")
    print(f"Final feature count: {X_all.shape[1]}")

    # =========================================================================
    # STEP 3: Split data (stratified by LOS tier)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Splitting data")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # =========================================================================
    # STEP 4: Train XGBoost multiclass classifier
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Training XGBoost (multiclass LOS prediction)")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1.0,
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        early_stopping_rounds=20,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # =========================================================================
    # STEP 5: Evaluate
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Evaluation")
    print("=" * 60)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=LOS_LABELS))

    wf1 = f1_score(y_val, y_pred, average='weighted')
    print(f"Weighted F1-Score: {wf1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=LOS_LABELS)
    disp.plot(cmap='Blues')
    plt.title(f'Model 5 — LOS Prediction (Weighted F1: {wf1:.4f})')
    plt.tight_layout()
    plt.savefig(SAVED_MODEL_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved.")

    # Feature importance (top 10)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top10 = importances.sort_values(ascending=False).head(10)
    print(f"\nTop 10 features for LOS prediction:")
    for feat, imp in top10.items():
        print(f"  {feat}: {imp:.4f}")

    # =========================================================================
    # STEP 6: Save model and artifacts
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Saving model")
    print("=" * 60)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / 'model.joblib')
    joblib.dump(preprocessing_state, SAVED_MODEL_DIR / 'preprocessing_state.joblib')
    joblib.dump(list(X_train.columns), SAVED_MODEL_DIR / 'feature_names.joblib')
    joblib.dump(wf1, SAVED_MODEL_DIR / 'metric_value.joblib')

    print(f"Model saved to {SAVED_MODEL_DIR / 'model.joblib'}")

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — Weighted F1: {wf1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()