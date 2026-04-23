#!/usr/bin/env python3
"""
Model 2: Deep Learning — Training Script
==========================================
TensorFlow/Keras DNN for hospital readmission prediction.
Same data as Model 1 — compare DNN vs XGBoost performance.

Usage: python models/model2_deep_learning/train.py
"""
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Add project root to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import prepare_training_data

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "patient_encounters_2023.csv"
SAVED_MODEL_DIR = Path(__file__).resolve().parent / "saved_model"


def train():
    """Full training pipeline: load data, scale, train DNN, evaluate, save."""

    # =========================================================================
    # STEP 1: Load and prepare data using shared pipeline
    # =========================================================================
    print("=" * 60)
    print("STEP 1: Loading and preparing data")
    print("=" * 60)
    X_train, X_val, y_train, y_val, preprocessing_state = prepare_training_data(str(DATA_PATH))

    # =========================================================================
    # STEP 2: Scale features (required for neural networks)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 2: Scaling features")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"Features scaled: {X_train_scaled.shape[1]} columns")

    # =========================================================================
    # STEP 3: Compute class weights for imbalance
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 3: Computing class weights for imbalance")
    print("=" * 60)

    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight = {0: cw[0], 1: cw[1]}
    print(f"Class weights: {class_weight}")

    # =========================================================================
    # STEP 4: Build DNN architecture
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 4: Building DNN architecture")
    print("=" * 60)

    input_dim = X_train_scaled.shape[1]

    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.51),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.59),

        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.55),

        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.00144),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    model.summary()

    # =========================================================================
    # STEP 5: Train with early stopping
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 5: Training DNN")
    print("=" * 60)

    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max',
        ),
        ReduceLROnPlateau(
            monitor='val_auc',
            mode='max',
            patience=7,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # =========================================================================
    # STEP 6: Evaluate on validation set
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 6: Evaluation on validation set")
    print("=" * 60)

    y_proba = model.predict(X_val_scaled).flatten()
    y_pred = (y_proba >= 0.5).astype(int)

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
    plt.title(f'Model 2 — DNN Confusion Matrix (AUC: {auc:.4f})')
    plt.tight_layout()
    plt.savefig(SAVED_MODEL_DIR / 'confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to saved_model/confusion_matrix.png")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['auc'], label='Train AUC')
    axes[1].plot(history.history['val_auc'], label='Val AUC')
    axes[1].set_title('AUC-ROC Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(SAVED_MODEL_DIR / 'training_curves.png', dpi=150)
    plt.close()
    print("Training curves saved to saved_model/training_curves.png")

    # =========================================================================
    # STEP 7: Save model, scaler, and preprocessing state
    # =========================================================================
    print("\n" + "=" * 60)
    print("STEP 7: Saving model and artifacts")
    print("=" * 60)

    SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(SAVED_MODEL_DIR / 'model.keras')
    joblib.dump(scaler, SAVED_MODEL_DIR / 'scaler.joblib')
    joblib.dump(preprocessing_state, SAVED_MODEL_DIR / 'preprocessing_state.joblib')
    joblib.dump(list(X_train.columns), SAVED_MODEL_DIR / 'feature_names.joblib')

    print(f"Model saved to {SAVED_MODEL_DIR / 'model.keras'}")
    print(f"Scaler saved to {SAVED_MODEL_DIR / 'scaler.joblib'}")
    print(f"Preprocessing state saved to {SAVED_MODEL_DIR / 'preprocessing_state.joblib'}")
    print(f"Feature names saved to {SAVED_MODEL_DIR / 'feature_names.joblib'}")

    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE — AUC-ROC: {auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    train()