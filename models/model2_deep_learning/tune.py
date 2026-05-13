#!/usr/bin/env python3
"""
Keras Tuner (Hyperband) for Model 2 DNN.
Searches architecture, regularization, and training hyperparameters.

Usage: python models/model2_deep_learning/tune.py
"""
import sys
import os
import numpy as np
from pathlib import Path

# Suppress TensorFlow info logs for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from pipelines.data_pipeline import prepare_training_data

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "patient_encounters_2023.csv"
TUNER_DIR = Path(__file__).resolve().parent / "tuner_results"

# =========================================================================
# Load and prepare data ONCE (outside the search loop)
# =========================================================================
print("Loading data for Keras Tuner...")
X_train, X_val, y_train, y_val, _ = prepare_training_data(str(DATA_PATH))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Class weights
cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
class_weight = {0: cw[0], 1: cw[1]}

input_dim = X_train_scaled.shape[1]
print(f"Input dimension: {input_dim} features")
print(f"Class weights: {class_weight}")


# =========================================================================
# Define the model builder for Keras Tuner
# =========================================================================
def build_model(hp):
    """
    Build a DNN with hyperparameters chosen by Keras Tuner.

    Search space designed based on baseline analysis:
    - Baseline overfits with 256->128->64 (train AUC 0.71, val AUC 0.685)
    - Focus on SMALLER architectures + STRONGER regularization
    - 39 features → ideal first layer is 64-128 (2:1 to 3:1 ratio)
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))

    # Number of hidden layers (2 to 4)
    n_layers = hp.Int('n_layers', min_value=2, max_value=4, step=1)

    # Use BatchNorm? (yes/no as hyperparameter)
    use_batchnorm = hp.Boolean('use_batchnorm')

    for i in range(n_layers):
        # Units per layer — DECREASE with depth (funnel shape)
        # First layer: 64-192, subsequent layers shrink
        if i == 0:
            units = hp.Int(f'units_L{i}', min_value=64, max_value=192, step=32)
        else:
            # Each subsequent layer is same or smaller than previous
            units = hp.Int(f'units_L{i}', min_value=32, max_value=128, step=32)

        model.add(Dense(units, activation='relu'))

        if use_batchnorm:
            model.add(BatchNormalization())

        # Dropout — search HIGHER values to combat overfitting
        dropout_rate = hp.Float(f'dropout_L{i}', min_value=0.2, max_value=0.6, step=0.05)
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='sigmoid'))

    # Learning rate — search in log scale, biased toward lower values
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=5e-3, sampling='log')

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    return model


# =========================================================================
# Manual Hyperband-style search (no keras-tuner dependency needed)
# =========================================================================
def run_search(n_trials=30, max_epochs=60):
    """
    Manual random search with early stopping.
    Avoids requiring keras-tuner package installation.
    Each trial trains a model with random hyperparameters and reports val AUC.
    """
    import random
    random.seed(42)

    best_auc = 0.0
    best_params = {}
    all_results = []

    print(f"\nStarting search ({n_trials} trials, max {max_epochs} epochs each)...")
    print("=" * 60)

    for trial in range(n_trials):
        # Sample hyperparameters
        n_layers = random.randint(2, 4)
        use_batchnorm = random.choice([True, False])
        learning_rate = 10 ** random.uniform(-4, np.log10(5e-3))
        batch_size = random.choice([32, 64, 128, 256])

        # Build architecture
        model = Sequential()
        model.add(Input(shape=(input_dim,)))

        layer_configs = []
        for i in range(n_layers):
            if i == 0:
                units = random.choice([64, 96, 128, 160, 192])
            else:
                units = random.choice([32, 64, 96, 128])
            dropout = round(random.uniform(0.2, 0.6), 2)
            layer_configs.append((units, dropout))

            model.add(Dense(units, activation='relu'))
            if use_batchnorm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
        )

        # Train with early stopping
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True,
                mode='max',
            ),
        ]

        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0,
        )

        # Evaluate
        y_proba = model.predict(X_val_scaled, verbose=0).flatten()
        auc = roc_auc_score(y_val, y_proba)
        best_epoch = np.argmax(history.history['val_auc']) + 1

        params = {
            'n_layers': n_layers,
            'layers': layer_configs,
            'use_batchnorm': use_batchnorm,
            'learning_rate': round(learning_rate, 6),
            'batch_size': batch_size,
            'best_epoch': best_epoch,
        }
        all_results.append((auc, params))

        # Track best
        marker = ""
        if auc > best_auc:
            best_auc = auc
            best_params = params
            marker = " ★ NEW BEST"

        layers_str = " → ".join([f"{u}(d={d})" for u, d in layer_configs])
        print(f"Trial {trial+1:2d}/{n_trials} | AUC: {auc:.4f} | "
              f"lr={learning_rate:.5f} bs={batch_size:3d} bn={str(use_batchnorm):5s} | "
              f"{layers_str} | ep={best_epoch}{marker}")

    # Print final results
    print("\n" + "=" * 60)
    print("SEARCH COMPLETE")
    print("=" * 60)

    # Top 5 results
    all_results.sort(key=lambda x: x[0], reverse=True)
    print("\nTop 5 trials:")
    for i, (auc, params) in enumerate(all_results[:5]):
        layers_str = " → ".join([f"{u}(d={d})" for u, d in params['layers']])
        print(f"  #{i+1}: AUC={auc:.4f} | lr={params['learning_rate']:.5f} "
              f"bs={params['batch_size']} bn={params['use_batchnorm']} | {layers_str}")

    print(f"\n{'='*60}")
    print(f"BEST AUC: {best_auc:.4f}")
    print(f"{'='*60}")
    print(f"\nBest Parameters:")
    print(f"    n_layers: {best_params['n_layers']}")
    print(f"    layers: {best_params['layers']}")
    print(f"    use_batchnorm: {best_params['use_batchnorm']}")
    print(f"    learning_rate: {best_params['learning_rate']}")
    print(f"    batch_size: {best_params['batch_size']}")
    print(f"    best_epoch: {best_params['best_epoch']}")
    print(f"\nCopy these parameters into your train.py!")

    return best_auc, best_params


if __name__ == "__main__":
    best_auc, best_params = run_search(n_trials=30, max_epochs=60)
    