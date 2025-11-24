import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import pathlib
import random

# Base directory = TestTensorFlow
base_dir = pathlib.Path(__file__).resolve().parent

# Model path
model_path = base_dir / "saved_models" / "face_classifier.keras"
model = keras.models.load_model(model_path)

# Directories for face and non_face images
face_dir = base_dir / "data" / "val" / "face"
non_face_dir = base_dir / "data" / "val" / "non_face"


def get_random_image_path(dir_path: pathlib.Path) -> pathlib.Path:
    # Collect jpg and png files
    files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
    if not files:
        raise RuntimeError(f"No images found in {dir_path}")
    return random.choice(files)


def predict_one_image(img_path: pathlib.Path) -> float:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (128, 128))

    # Do not divide by 255 here, model already has Rescaling(1/255)
    input_tensor = np.expand_dims(img_resized, axis=0).astype("float32")

    pred = model.predict(input_tensor)[0][0]
    return float(pred)


def run_once():
    # Randomly pick one image from each class
    samples = [
        ("face", get_random_image_path(face_dir)),
        ("non_face", get_random_image_path(non_face_dir)),
    ]

    for true_label, img_path in samples:
        pred = predict_one_image(img_path)

        # pred ≈ P(non_face)
        if pred >= 0.5:
            predicted_label = "non_face"
            result = "non-face"
        else:
            predicted_label = "face"
            result = "face"

        correctness = "Correct" if predicted_label == true_label else "Wrong"

        print("==========")
        print(f"Image source: {true_label}")
        print(f"Filename: {img_path.name}")
        print(f"Raw prediction: {pred}")
        print(f"Prediction result: {result}, {correctness}")
    print("==========")


if __name__ == "__main__":
    run_once()
