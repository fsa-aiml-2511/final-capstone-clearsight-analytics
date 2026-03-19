#!/usr/bin/env python3
"""
Model 4: NLP Classification — Prediction Script
=================================================
Loads your trained model and generates predictions on test data.

Usage: python predict.py
Output: test_data/model4_results.csv
"""
import pandas as pd
from pathlib import Path

# Paths
MODEL_PATH = Path("models/model4_nlp_classification/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model4_results.csv"


def load_model():
    """Load your trained NLP model from saved_model/.

    Typical approaches:
        # Scikit-learn pipeline with TF-IDF + classifier
        import joblib
        model = joblib.load(MODEL_PATH / "model.joblib")

        # Hugging Face transformer
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    Don't forget to load your tokenizer / vectorizer if needed.
    """
    # TODO: Load your saved model
    raise NotImplementedError("Load your trained model here")


def preprocess_text(texts):
    """Apply text preprocessing: lowercasing, tokenization, cleaning, etc."""
    # TODO: Apply the same preprocessing used during training
    raise NotImplementedError("Preprocess text here")


def predict(model, test_data):
    """Generate predictions on text data.

    Should return a DataFrame with columns: id, predicted_class, confidence
    """
    # TODO: Run your model on the test data
    raise NotImplementedError("Generate predictions here")


def main():
    # Load model
    model = load_model()

    # Load test data
    # TODO: Update this path to match your test data file
    # test_df = pd.read_csv(TEST_DATA_DIR / "test_data_file.csv")

    # Preprocess text
    # processed = preprocess_text(test_df["text_column"])

    # Generate predictions
    # predictions = predict(model, processed)

    # Save results — MUST match output template exactly
    # results = pd.DataFrame({
    #     "id": test_df["id"],
    #     "predicted_class": predicted_classes,
    #     "confidence": confidence_scores,
    # })
    # results.to_csv(OUTPUT_FILE, index=False)

    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
