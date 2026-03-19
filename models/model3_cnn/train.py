#!/usr/bin/env python3
"""
Model 3: CNN — Training Script
================================
Train a convolutional neural network for image classification.
Transfer learning is recommended (ResNet50, EfficientNet, DenseNet).

Framework: TensorFlow / Keras

IMPORTANT: Resize images before training! Raw images may be very high resolution
and will cause memory errors if loaded full-size.
"""
from pathlib import Path

RAW_IMAGES = Path("data/raw/")
SAVED_MODEL_DIR = Path("models/model3_cnn/saved_model/")


def load_images(image_dir, target_size=(224, 224)):
    """Load and preprocess images from directory.

    Example using Keras ImageDataGenerator:
        from tensorflow.keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            horizontal_flip=True,
            zoom_range=0.2,
        )
        train_gen = datagen.flow_from_directory(
            image_dir,
            target_size=target_size,
            batch_size=32,
            class_mode='binary',
            subset='training',
        )

    IMPORTANT: Handle class imbalance with class_weight or augmentation.
    """
    # TODO: Load and preprocess your images
    raise NotImplementedError


def build_model():
    """Build or fine-tune a CNN.

    Transfer learning example:
        import tensorflow as tf

        base_model = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3)
        )
        base_model.trainable = False  # Freeze base layers initially

        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    """
    # TODO: Build your CNN
    raise NotImplementedError


def train_model(model, train_data, val_data):
    """Train the CNN with callbacks.

    Use EarlyStopping and optionally ReduceLROnPlateau.
    Pass class_weight to model.fit() to handle imbalance.
    """
    # TODO: Train your model
    raise NotImplementedError


def evaluate_model(model, val_data):
    """Evaluate CNN performance.

    Must include:
    - Accuracy and weighted F1
    - Confusion matrix
    - Sample predictions with images

    Bonus: Grad-CAM visualizations showing what the model "sees"
    """
    # TODO: Evaluate your model
    raise NotImplementedError


def save_model(model):
    """Save the trained model.

    Example:
        SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save(SAVED_MODEL_DIR / "model.keras")
    """
    # TODO: Save your model
    raise NotImplementedError


def main():
    # 1. Load and preprocess images
    # train_data, val_data = load_images(RAW_IMAGES / "images")

    # 2. Build model
    # model = build_model()

    # 3. Train
    # train_model(model, train_data, val_data)

    # 4. Evaluate
    # evaluate_model(model, val_data)

    # 5. Save
    # save_model(model)

    print("Training complete!")


if __name__ == "__main__":
    main()
