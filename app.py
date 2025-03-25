import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from classifier import classify

app = Flask(__name__)
app.config["STATIC_FOLDER"] = "static"
app.config["UPLOAD_FOLDER"] = os.path.join(
    app.config["STATIC_FOLDER"], "uploads"
)

cnn_model = tf.keras.models.load_model(
    os.path.join(app.config["STATIC_FOLDER"], "models", "f16_b1_c130_mi24.h5")
)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify_endpoint():
    file = request.files.get("image")
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        label, prob = classify(cnn_model, filepath)
        return jsonify({"label": label, "prob": prob})
    return jsonify({"error": "No file uploaded"}), 400


if __name__ == "__main__":
    app.run(debug=True)
