from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
from feature_extraction import extract_features

app = Flask(__name__, template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts"))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

model = joblib.load(os.path.join(ARTIFACTS_DIR, "quality_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    image_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(image_path)

    features = extract_features(image_path)
    features = scaler.transform(features)

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    result = "GOOD QUALITY IMAGE" if pred == 1 else "BAD QUALITY IMAGE"
    confidence = round(float(np.max(prob)) * 100, 2)

    return jsonify({
        "result": result,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
