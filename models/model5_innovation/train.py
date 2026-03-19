#!/usr/bin/env python3
"""
Model 5: Innovation — Training Script
=======================================
This is your team's innovation model. Identify a problem in the data that
Models 1-4 don't address, and build a model that solves it.

Requirements:
- Clear value proposition (why does this model matter?)
- Defined success metric (you choose what to optimize)
- Cost-benefit estimate (what's the ROI?)

Use whatever approach fits your problem best — traditional ML, deep learning,
clustering, anomaly detection, time series, etc.
"""
from pathlib import Path

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model5_innovation/saved_model/")


def load_data():
    """Load data for your innovation model."""
    # TODO: Load your dataset
    raise NotImplementedError


def preprocess(df):
    """Preprocess data for your chosen problem."""
    # TODO: Prepare features
    raise NotImplementedError


def train_model(X_train, y_train):
    """Train your innovation model."""
    # TODO: Train your model
    raise NotImplementedError


def evaluate_model(model, X_val, y_val):
    """Evaluate with your custom success metric.

    Must include:
    - Your chosen metric (and why you chose it)
    - Baseline comparison (what would a naive approach get?)
    - Business impact estimate
    """
    # TODO: Evaluate your model
    raise NotImplementedError


def save_model(model):
    """Save your model to saved_model/.

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

    # 2. Preprocess
    # X_train, X_val, y_train, y_val = preprocess(df)

    # 3. Train
    # model = train_model(X_train, y_train)

    # 4. Evaluate
    # evaluate_model(model, X_val, y_val)

    # 5. Save
    # save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
