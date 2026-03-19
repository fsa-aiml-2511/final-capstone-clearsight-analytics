#!/usr/bin/env python3
"""
Model 2: Deep Learning — Training Script
==========================================
Train a deep neural network on the same tabular data as Model 1.
Compare performance against your traditional ML model.

Framework: TensorFlow / Keras
"""
from pathlib import Path

PROCESSED_DATA = Path("data/processed/")
SAVED_MODEL_DIR = Path("models/model2_deep_learning/saved_model/")


def load_data():
    """Load preprocessed data from data/processed/.

    Use the shared pipeline:
        from pipelines.data_pipeline import load_processed_data
        df = load_processed_data()
    """
    # TODO: Load your preprocessed dataset
    raise NotImplementedError


def preprocess_features(df):
    """Prepare features for neural network training.

    DNN-specific considerations:
    - Scale all features to [0,1] or standardize (mean=0, std=1)
    - One-hot encode categoricals (or use embedding layers)
    - Convert to numpy arrays or tf.data.Dataset
    """
    # TODO: Prepare your feature matrix X and target y
    raise NotImplementedError


def build_model(input_dim, num_classes):
    """Define your neural network architecture.

    Example:
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )
        return model

    IMPORTANT: For class imbalance, use class_weight parameter in model.fit()
    or use a weighted loss function.
    """
    # TODO: Build your neural network
    raise NotImplementedError


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model with early stopping.

    Example:
        from tensorflow.keras.callbacks import EarlyStopping

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            class_weight=class_weights,  # Handle imbalance!
        )
    """
    # TODO: Train your model
    raise NotImplementedError


def evaluate_model(model, X_val, y_val):
    """Evaluate and compare against Model 1.

    Must include:
    - Classification report
    - Weighted F1 score
    - Training curves (loss and accuracy over epochs)
    - Comparison table: Model 1 vs Model 2 metrics
    """
    # TODO: Print evaluation metrics
    raise NotImplementedError


def save_model(model):
    """Save the trained model to saved_model/.

    Example:
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(SAVED_MODEL_DIR / "model.keras")
    """
    # TODO: Save your model
    raise NotImplementedError


def main():
    # 1. Load data
    df = load_data()

    # 2. Preprocess features
    # X_train, X_val, y_train, y_val = preprocess_features(df)

    # 3. Build model
    # model = build_model(input_dim=X_train.shape[1], num_classes=4)

    # 4. Train
    # train_model(model, X_train, y_train, X_val, y_val)

    # 5. Evaluate and compare to Model 1
    # evaluate_model(model, X_val, y_val)

    # 6. Save
    # save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
