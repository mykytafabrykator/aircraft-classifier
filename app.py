import os

import tensorflow as tf
from flask import Flask, request

from classifier import classify


app = Flask(__name__)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = "static/uploads/"

cnn_model = tf.keras.models.load_model(
    STATIC_FOLDER + "/models/" + "f16_b1_c130_mi24.h5"
)


@app.route("/")
def home():
    return "Use '/classify' endpoint to classify an image."


@app.route("/classify", methods=["POST"])
def upload_file():
    file = request.files["image"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)

    label, prob = classify(cnn_model, upload_image_path)

    return {
        "label": label,
        "probability": prob,
    }


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)
