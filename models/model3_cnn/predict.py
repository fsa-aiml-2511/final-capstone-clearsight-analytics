import pandas as pd
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
# Paths
MODEL_PATH = Path("models/model3_cnn/saved_model/")
TEST_DATA_DIR = Path("test_data/")
OUTPUT_FILE = TEST_DATA_DIR / "model3_results.csv"


# ---------------------------------------------------------
# 1. Load trained model (Keras 3 compatible)
# ---------------------------------------------------------
def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH / "best_model.keras",
        compile=False
    )
    print("Model loaded.")
    return model

# ---------------------------------------------------------
# 2. Load + preprocess test images (224×224)
# ---------------------------------------------------------
def load_images(image_paths):
    images = []          
    valid_ids = []       

    for img_id, path in image_paths.items():
        try:
            img = tf.keras.utils.load_img(path, target_size=(224, 224))
        except Exception as e:
            print(f"WARNING: Could not load {path}: {e}")
            continue

        img = tf.keras.utils.img_to_array(img)
        img = preprocess_input(img)

        images.append(img)
        valid_ids.append(img_id)

    return np.array(images), valid_ids   


# ---------------------------------------------------------
# 3. Generate predictions
# ---------------------------------------------------------
def predict(model, images, image_ids):
    preds = model(images).numpy()

    predicted_classes = (preds > 0.5).astype(int).flatten()
    confidence_scores = np.where(preds.flatten() > 0.5, preds.flatten(), 1 - preds.flatten())

    df = pd.DataFrame({
        "image_id": image_ids,
        "predicted_class": predicted_classes,
        "confidence": confidence_scores,
    })

    return df

# ---------------------------------------------------------
# 4. Main pipeline
# ---------------------------------------------------------
def main():
    # Load model
    model = load_model()

    # Load test images
    test_image_dir = TEST_DATA_DIR / "images"
    VALID_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    image_paths = {
        img_path.name: img_path
        for img_path in sorted(test_image_dir.glob("*.*"))
        if img_path.suffix.lower() in VALID_EXTENSIONS
    }

    images, image_ids = load_images(image_paths)

    # Generate predictions
    results = predict(model, images, image_ids)

    # Save results
    results.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

   
    