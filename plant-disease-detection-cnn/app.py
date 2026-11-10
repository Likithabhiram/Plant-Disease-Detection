import json
import uuid
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from config import (
    MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE, ALLOWED_EXTENSIONS,
    MAX_UPLOAD_MB, BASE_DIR
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

model = None
class_names = []


def load_artifacts():
    global model, class_names

    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Trained model files are missing. Run `python train.py` before starting the app."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def readable_label(label):
    return label.replace("___", " - ").replace("_", " ")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            error="Model is not loaded. Train the model and restart the app."
        )

    uploaded_file = request.files.get("leaf_image")
    if not uploaded_file or uploaded_file.filename == "":
        return render_template("index.html", error="Please choose a leaf image.")

    if not allowed_file(uploaded_file.filename):
        return render_template(
            "index.html",
            error="Unsupported file type. Upload PNG, JPG, JPEG or WEBP."
        )

    safe_name = secure_filename(uploaded_file.filename)
    extension = Path(safe_name).suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / unique_name

    try:
        uploaded_file.save(saved_path)
        image_batch = prepare_image(saved_path)
        probabilities = model.predict(image_batch, verbose=0)[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index]) * 100

        top_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = [
            {
                "label": readable_label(class_names[int(index)]),
                "confidence": float(probabilities[index]) * 100,
            }
            for index in top_indices
        ]

        return render_template(
            "result.html",
            image_url=f"/static/uploads/{unique_name}",
            prediction=readable_label(predicted_class),
            confidence=confidence,
            top_predictions=top_predictions,
        )
    except (UnidentifiedImageError, OSError, ValueError):
        if saved_path.exists():
            saved_path.unlink()
        return render_template(
            "index.html",
            error="The uploaded file could not be read as an image. Try another image."
        )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        error=f"File is too large. Maximum upload size is {MAX_UPLOAD_MB} MB."
    ), 413


if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
