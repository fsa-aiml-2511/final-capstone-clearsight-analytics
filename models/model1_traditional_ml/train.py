#!/usr/bin/env python3
"""
Model 1: Traditional ML — Training Script
===========================================
Train a classical ML model (XGBoost, Random Forest, etc.) on your scenario's
tabular data.

IMPORTANT: This model must be interpretable. Include SHAP or feature importance
analysis so stakeholders can understand WHY the model makes its predictions.
"""
from pathlib import Path

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model1_traditional_ml/saved_model/")


def load_data():
    """Load preprocessed data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    # TODO: Load your preprocessed dataset
    raise NotImplementedError


def preprocess_features(df):
    """Select and prepare features for training.

    Consider:
    - Feature selection (drop leaky or irrelevant columns)
    - Encoding categorical variables
    - Scaling numerical features
    - Handling missing values
    """
    # TODO: Prepare your feature matrix X and target y
    raise NotImplementedError


def train_model(X_train, y_train):
    """Train your traditional ML model.

    Recommended algorithms:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from xgboost import XGBClassifier

    IMPORTANT: Handle class imbalance!
        model = RandomForestClassifier(class_weight='balanced')
    """
    # TODO: Train your model
    raise NotImplementedError


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance on validation data.

    Must include:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix
    - Weighted F1 score (primary metric for imbalanced data)
    - AUC-ROC (for binary classification scenarios)
    """
    # TODO: Print evaluation metrics
    raise NotImplementedError


def explain_model(model, X_val):
    """Generate SHAP or feature importance analysis.

    This is REQUIRED — your model must be interpretable.

    Option 1 — SHAP (recommended):
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val)
        shap.summary_plot(shap_values, X_val)

    Option 2 — Built-in feature importance:
        importances = model.feature_importances_
        # Plot top 15 features
    """
    # TODO: Generate explainability analysis
    raise NotImplementedError


def save_model(model):
    """Save the trained model to saved_model/.

    Example:
        import joblib
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
    """
    # TODO: Save your model
    raise NotImplementedError


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess features
    # X_train, X_val, y_train, y_val = preprocess_features(df)

    # 3. Train model
    # model = train_model(X_train, y_train)

    # 4. Evaluate
    # evaluate_model(model, X_val, y_val)

    # 5. Explain — REQUIRED
    # explain_model(model, X_val)

    # 6. Save
    # save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
