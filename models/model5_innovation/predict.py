#!/usr/bin/env python3
"""
Model 5: Innovation — Prediction Script
=========================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model5_results.csv
"""
import pandas as pd
from pathlib import Path

# Paths
MODEL_PATH = Path("models/model5_innovation/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model5_results.csv"


def load_model():
    """Load your trained model from saved_model/.

    This is your team's innovation model — use whatever approach you chose.
    """
    # TODO: Load your saved model
    raise NotImplementedError("Load your trained model here")


def predict(model, test_data):
    """Generate predictions on test data.

    Should return a DataFrame with columns:
        id, prediction, confidence, metric_name, metric_value

    metric_name and metric_value are your custom evaluation metric.
    For example: metric_name="f1_weighted", metric_value=0.85
    """
    # TODO: Run your model on the test data
    raise NotImplementedError("Generate predictions here")


def main():
    # Load model
    model = load_model()

    # Load test data
    # TODO: Update this path to match your test data file
    # test_df = pd.read_csv(TEST_DATA_DIR / "test_data_file.csv")

    # Generate predictions
    # predictions = predict(model, test_df)

    # Save results — MUST match output template exactly
    # results = pd.DataFrame({
    #     "id": test_df["id"],
    #     "prediction": predictions,
    #     "confidence": confidence_scores,
    #     "metric_name": "your_custom_metric",
    #     "metric_value": metric_score,
    # })
    # results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
