#!/usr/bin/env python3
"""
Model 4: NLP Classification — Training Script
===============================================
Train a text classification model on your scenario's text data.

Approaches (pick one):
- TF-IDF + traditional classifier (simplest, often surprisingly good)
- LSTM / GRU neural network
- Fine-tuned transformer (BERT, DistilBERT)

IMPORTANT: Save your vectorizer/tokenizer alongside the model — you'll need
the same text preprocessing at prediction time.
"""
from pathlib import Path

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model4_nlp_classification/saved_model/")


def load_data():
    """Load text data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    # TODO: Load your text dataset
    raise NotImplementedError


def preprocess_text(texts):
    """Clean and prepare text for modeling.

    Common steps:
    - Lowercase
    - Remove special characters, HTML tags
    - Handle abbreviations and slang
    - Tokenize
    - Remove stopwords (optional — sometimes they help)

    IMPORTANT: Apply the SAME preprocessing at prediction time.
    """
    # TODO: Clean your text data
    raise NotImplementedError


def vectorize_text(texts):
    """Convert text to numerical features.

    Option 1 — TF-IDF (simplest):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        # Save vectorizer! You need it at prediction time.
        joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")

    Option 2 — Embeddings (for neural network approaches):
        from tensorflow.keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(texts)
    """
    # TODO: Vectorize your text
    raise NotImplementedError


def train_model(X_train, y_train):
    """Train your text classifier.

    TF-IDF approach:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X_train, y_train)

    Neural network approach:
        import tensorflow as tf
        model = tf.keras.Sequential([...])
    """
    # TODO: Train your model
    raise NotImplementedError


def evaluate_model(model, X_val, y_val):
    """Evaluate NLP model performance.

    Must include:
    - Classification report per category
    - Weighted F1 score
    - Confusion matrix
    - Example predictions with actual text
    """
    # TODO: Evaluate your model
    raise NotImplementedError


def save_model(model):
    """Save model AND vectorizer/tokenizer.

    IMPORTANT: You must save both the model and the text preprocessor.

    Example:
        import joblib
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
        joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")
    """
    # TODO: Save your model and vectorizer
    raise NotImplementedError


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess text
    # texts = preprocess_text(df["text_column"])

    # 3. Vectorize
    # X = vectorize_text(texts)

    # 4. Split (use stratified split for imbalanced classes)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y)

    # 5. Train
    # model = train_model(X_train, y_train)

    # 6. Evaluate
    # evaluate_model(model, X_val, y_val)

    # 7. Save model + vectorizer
    # save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
