#!/usr/bin/env python3
"""
Optuna hyperparameter tuning for XGBoost
Finds the best parameters to break the 0.70 AUC barrier.
"""
import sys
from pathlib import Path
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import prepare_training_data

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "patient_encounters_2023.csv"

# Load data once (outside the objective function so it's fast)
print("Loading data for Optuna tuning...")
X_train, X_val, y_train, y_val, _ = prepare_training_data(str(DATA_PATH))

# Calculate scale_pos_weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

def objective(trial):
    """
    Optuna objective function for XGBoost with expanded ranges.

    Ranges expanded vs first tuning run to account for new feature space
    (6 additional target-encoded features are continuous, while the ones
    they replaced were discrete LabelEncoded integers). This changes the
    optimal splits and regularization pressure.
    """
    param = {
        'n_estimators': 1500,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 8.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5.0),
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'auc',
        'random_state': 42,
        'early_stopping_rounds': 25,
    }

    model = XGBClassifier(**param)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)

    return auc

if __name__ == "__main__":
    # Create study
    study = optuna.create_study(direction="maximize")
    
    print("\nStarting Optuna study (80 trials — expanded search space)...")
    study.optimize(objective, n_trials=80)
    
    # Print results
    print("\n" + "="*50)
    print("BEST TRIAL FOUND")
    print("="*50)
    print(f"Best AUC: {study.best_value:.4f}")
    print("\nBest Parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    print("\nCopy these parameters into your train.py!")