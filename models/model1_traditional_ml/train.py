#!/usr/bin/env python3
"""
Model 1: Traditional ML — Training Script
===========================================
XGBoost classifier for hospital readmission prediction.
Uses scale_pos_weight for class imbalance and SHAP for interpretability.
Usage: python models/model1_traditional_ml/train.py
"""
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import shap
import pandas as pd

# Add project root to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import prepare_training_data

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "patient_encounters_2023.csv"
SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"


def train():
    """Full training pipeline: load data, train XGBoost with scale_pos_weight, evaluate, SHAP, save."""

    # =========================================================================
    # STEP 1: Load and prepare data using shared pipeline
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Loading and preparing data")
    print("=" * 60)
    X_train, X_val, y_train, y_val, preprocessing_state = prepare_training_data(str(DATA_PATH))

    # =========================================================================
    # STEP 2: Compute scale_pos_weight for class imbalance
    # =========================================================================
    # Replaces SMOTE: instead of synthesizing minority samples (which can add
    # noise in ambiguous clinical regions), we weight the loss function to
    # penalize misclassifying the minority class more heavily.
    # Formula: scale_pos_weight = count(negative) / count(positive)
    print("\n" + "=" * 60)
    print("STEP 2: Computing scale_pos_weight for class imbalance")
    print("=" * 60)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Class distribution: {{0: {neg_count}, 1: {pos_count}}}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    # =========================================================================
    # STEP 3: Train XGBoost with Early Stopping
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Training XGBoost")
    print("=" * 60)

    model = XGBClassifier(
        n_estimators=1500,
        max_depth=9,
        min_child_weight=1,
        learning_rate=0.008,
        subsample=0.84,
        colsample_bytree=0.64,
        gamma=0.91,
        reg_alpha=4.31,
        reg_lambda=0.64,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=25,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # =========================================================================
    # STEP 4: Evaluate on validation set
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation on validation set")
    print("=" * 60)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Not Readmitted', 'Readmitted']))

    # AUC-ROC
    auc = roc_auc_score(y_val, y_proba)
    print(f"AUC-ROC: {auc:.4f}")

    if auc >= 0.80:
        print(">>> STRETCH GOAL REACHED!")
    elif auc >= 0.70:
        print(">>> Minimum benchmark passed.")
    else:
        print(">>> WARNING: Below minimum benchmark of 0.70")

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Readmitted', 'Readmitted'])
    disp.plot(cmap='Blues')
    plt.title(f'Model 1 — XGBoost Confusion Matrix (AUC: {auc:.4f})')
    plt.tight_layout()
    plt.savefig(SAVED_MODEL_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to saved_model/confusion_matrix.png")

    # =========================================================================
    # STEP 5: SHAP Interpretability (REQUIRED)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: SHAP Feature Importance (Required)")
    print("=" * 60)

    explainer = shap.TreeExplainer(model)
    # Use a sample for speed (SHAP on 20k rows is slow)
    X_sample = X_val.sample(n=min(2000, len(X_val)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot (top 15 features)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SAVED_MODEL_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved to saved_model/shap_summary.png")

    # Bar plot (feature importance ranking)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(SAVED_MODEL_DIR / 'shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP bar plot saved to saved_model/shap_bar.png")

    # =========================================================================
    # STEP 6: Save model and preprocessing state
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Saving model and preprocessing state")
    print("=" * 60)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, SAVED_MODEL_DIR / 'model.joblib')
    joblib.dump(preprocessing_state, SAVED_MODEL_DIR / 'preprocessing_state.joblib')
    joblib.dump(list(X_train.columns), SAVED_MODEL_DIR / 'feature_names.joblib')

    print(f"Model saved to {SAVED_MODEL_DIR / 'model.joblib'}")
    print(f"Preprocessing state saved to {SAVED_MODEL_DIR / 'preprocessing_state.joblib'}")
    print(f"Feature names saved to {SAVED_MODEL_DIR / 'feature_names.joblib'}")

   
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — AUC-ROC: {auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()