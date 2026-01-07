from flask import Flask, request, jsonify
import os
import joblib
from feature_extraction import extract_features

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts"))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load trained model
model = joblib.load(os.path.join(ARTIFACTS_DIR, "quality_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

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
    label = "Good Image" if pred == 1 else "Bad Image"

    return jsonify({
        "prediction": label
    })

if __name__ == "__main__":
    app.run(debug=True)
