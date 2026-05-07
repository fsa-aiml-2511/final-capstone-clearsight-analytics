from pathlib import Path
from pyexpat import model
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os

RAW_IMAGES = Path("data/raw/healthcare_retinal_images/")
SAVED_MODEL_DIR = Path("models/model3_cnn/saved_model/")
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load + preprocess images

# Load labels and convert to binary
labels = pd.read_csv("data/raw/retinal_labels.csv")
labels["dr_binary"] = (labels["diagnosis"] > 0).astype(str)
labels["filename"] = labels["id_code"] + ".png"

# Filter: keep only labels where the image file exists
existing = set(os.listdir("data/raw/healthcare_retinal_images/"))
labels = labels[labels["filename"].isin(existing)]
print(f"Matched images: {len(labels)}")
print(labels["dr_binary"].value_counts())

# Use flow_from_dataframe
datagen = ImageDataGenerator(
preprocessing_function=preprocess_input,
validation_split=0.2,
rotation_range=20,
horizontal_flip=True,
zoom_range=0.2,
)

train_gen = datagen.flow_from_dataframe(
labels,
directory="data/raw/healthcare_retinal_images/",
x_col="filename",
y_col="dr_binary",
target_size=(224, 224),
batch_size=32,
class_mode="binary",
subset="training",
shuffle=True,
)

val_gen = datagen.flow_from_dataframe(
    labels,
    directory="data/raw/healthcare_retinal_images/",
    x_col="filename",
    y_col="dr_binary",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False,
)

# 2. Build ResNet50 transfer-learning model

def build_resnet50_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model

# 3. Train model with callbacks

def train_model():
    # train_gen and val_gen already created above
    global train_gen, val_gen

    model = build_resnet50_model()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, mode="max", monitor="val_auc"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint(
            filepath=str(SAVED_MODEL_DIR / "best_model.keras"),
            save_best_only=True
        )
    ]

    # Compute class weights
    classes = np.array([0, 1])
    cw = compute_class_weight("balanced", classes=classes, y=train_gen.classes)
    class_weight = {0: cw[0], 1: cw[1]}

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks,
        class_weight=class_weight
    )

    model.save(SAVED_MODEL_DIR / "model.keras")

    return model, history, val_gen


# 4. Evaluation Suite

def evaluate_model(model, val_gen, num_samples_to_show=9):

    val_gen.reset() 

    # Load all validation data safely
    images = []
    labels = []

    for i in range(len(val_gen)):
        x, y = val_gen[i]
        images.append(x)
        labels.append(y)

    images = np.concatenate(images)
    labels = np.concatenate(labels)

    preds = model.predict(images)
    y_pred = (preds > 0.5).astype(int).ravel()
    y_true = labels.astype(int)

    # Metrics
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1:", f1_score(y_true, y_pred, average="weighted"))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.show()

    # Sample predictions
    idxs = np.random.choice(len(images), size=min(num_samples_to_show, len(images)), replace=False)

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(idxs):
        plt.subplot(3, 3, i + 1)
        img = images[idx]

    # Safe normalization
        den = img.max() - img.min()
        if den > 0:
            img_norm = (img - img.min()) / den
        else:
            img_norm = img

        plt.imshow(img_norm)
        plt.title(f"T: {y_true[idx]}  P: {y_pred[idx]}")
        plt.axis("off")

    plt.suptitle("Sample Predictions")
    plt.show()

   # 5. Grad-CAM utilities

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap.numpy()


def save_gradcam(image_path, heatmap, output_path="gradcam_output.jpg", alpha=0.4):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    cv2.imwrite(output_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    print(f"Grad-CAM saved to {output_path}")


# 6. Main Execution

if __name__ == "__main__":
    model, history, val_gen = train_model()
    evaluate_model(model, val_gen)

test_image = "data/raw/class1/example.jpg"
img = load_img(test_image, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
heatmap = make_gradcam_heatmap(img_array, model)
save_gradcam(test_image, heatmap, output_path="gradcam_example.jpg")

print("Training complete! Evaluation + Grad-CAM generated.")