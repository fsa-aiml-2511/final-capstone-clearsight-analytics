#!/usr/bin/env python3
"""
Model 4: NLP Classification — Training Script
===============================================
Train a text classification model on patient medication feedback.
"""

APPROACH = "tfidf"
# APPROACH = "embed"
if APPROACH == "tfidf":
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

else:
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

import sys
import pandas as pd
from pathlib import Path
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipelines.data_pipeline_wes import load_raw_data, clean_data, engineer_features

SAVED_MODEL_DIR = PROJECT_ROOT / "models" / "model4_nlp_classification" / "saved_model"
TARGET_COL = "effectiveness_3class"

def load_data() -> pd.DataFrame:
    """Load and clean the patient medication feedback dataset via the shared pipeline."""
    df = load_raw_data("patient_medication_feedback.csv")
    df = clean_data(df)
    df = engineer_features(df)
    return df


def preprocess_text(texts):
    """Clean and prepare text for modeling.

    Returns a list of cleaned strings ready for vectorization.
    Apply the SAME function at prediction time.
    """
    import re

    cleaned = []
    for text in texts:
        text = str(text).lower()
        text = text.replace("\n", " ")
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append(text)
    return cleaned


def vectorize_text(texts, max_words=10000, max_len=200):
    """Convert text to numerical features using the selected APPROACH.

    Returns (X, preprocessor) where preprocessor is the vectorizer or tokenizer.
    """
    if APPROACH == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_words, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)

        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, SAVED_MODEL_DIR / "vectorizer.joblib")
        print(f"TF-IDF matrix: {X.shape}")
        return X, vectorizer

    else:  # embed
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")

        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(tokenizer, SAVED_MODEL_DIR / "tokenizer.joblib")
        print(f"Embedding matrix: {X.shape}")
        return X, tokenizer


def train_model(X_train, y_train, num_classes=3, vocab_size=10000, max_len=200):
    """Train a classifier using the selected APPROACH.

    Returns model for tfidf, or (model, label_encoder) for embed.
    """
    if APPROACH == "tfidf":
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        print("Logistic Regression training complete.")
        return model

    else:  # embed
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        y_cat = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len),
            tf.keras.layers.SpatialDropout1D(0.2),
            tf.keras.layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ])

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        model.fit(
            X_train, y_cat,
            epochs=10,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=1,
        )
        print("LSTM training complete.")
        return model, le


def evaluate_model(model, X_val, y_val, texts_val=None, le=None):
    """Evaluate model performance using the selected APPROACH."""

    if APPROACH == "tfidf":
        y_pred = model.predict(X_val)
        classes = model.classes_
        y_true = y_val

    else:  # embed
        y_prob = model.predict(X_val, verbose=0)
        y_pred = le.inverse_transform(y_prob.argmax(axis=1))
        y_true = y_val
        classes = le.classes_

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    print(f"Weighted F1 Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix — {APPROACH.upper()}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

    if texts_val is not None:
        print("\n--- Example Predictions ---")
        for text, actual, pred in zip(list(texts_val)[:5], list(y_true)[:5], list(y_pred)[:5]):
            print(f"  Text:      {text[:80]}...")
            print(f"  Actual:    {actual}")
            print(f"  Predicted: {pred}\n")


def save_model(model, preprocessor):
    """Save model and vectorizer/tokenizer to saved_model/."""
    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if APPROACH == "tfidf":
        joblib.dump(model, SAVED_MODEL_DIR / "model.joblib")
        joblib.dump(preprocessor, SAVED_MODEL_DIR / "vectorizer.joblib")
        print(f"Saved model and vectorizer to {SAVED_MODEL_DIR}")

    else:  # embed
        model.save(SAVED_MODEL_DIR / "model.keras")
        joblib.dump(preprocessor, SAVED_MODEL_DIR / "tokenizer.joblib")
        print(f"Saved model and tokenizer to {SAVED_MODEL_DIR}")


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess text
    texts = preprocess_text(df["benefitsReview"])
    y = df[TARGET_COL]

    # 3. Vectorize
    X, preprocessor = vectorize_text(texts)

    # 4. Split (texts split alongside X/y so example predictions show actual review text)
    X_train, X_val, y_train, y_val, _, texts_val = train_test_split(
        X, y, texts, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Train
    result = train_model(X_train, y_train)

    # 6. Evaluate
    if APPROACH == "tfidf":
        model = result
        evaluate_model(model, X_val, y_val, texts_val=texts_val)
    else:
        model, le = result
        evaluate_model(model, X_val, y_val, texts_val=texts_val, le=le)

    # 7. Save
    save_model(model, preprocessor)

    print("Training complete!")


if __name__ == "__main__":
    main()
