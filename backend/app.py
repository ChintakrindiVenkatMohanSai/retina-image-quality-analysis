import os
import joblib
from flask import Flask, render_template, request, send_from_directory
from feature_extraction import extract_features

app = Flask(__name__)

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
MODEL_NAME = "Logistic Regression (RFE – 15 Quality Features)"

MODEL_PATH = os.path.join(
    ARTIFACTS_DIR,
    "logreg_rfe_best_by_test.pkl"
)

print("Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)

# =========================
# FEATURE NAMES (ORDER MATTERS)
# =========================
FEATURE_NAMES = [
    "Mean", "Std", "Skewness", "Entropy", "Median",
    "Contrast_0", "Correlation_0", "Homogeneity_0",
    "Contrast_45", "Energy_45", "Homogeneity_45",
    "Energy_90", "Homogeneity_90",
    "Contrast_135", "Homogeneity_135"
]

# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None
    extracted_features = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filename = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            # -------------------------
            # FEATURE EXTRACTION
            # -------------------------
            features = extract_features(image_path)

            # -------------------------
            # MODEL INFERENCE
            # -------------------------
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0, 1] * 100

            result = "GOOD QUALITY IMAGE" if prediction == 1 else "BAD QUALITY IMAGE"
            confidence = round(probability, 2)

            # -------------------------
            # SEND IMAGE & FEATURES TO UI
            # -------------------------
            image_url = f"/uploads/{filename}"

            extracted_features = dict(
                zip(FEATURE_NAMES, features.flatten().round(4))
            )

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        extracted_features=extracted_features,
        model_name=MODEL_NAME
    )

# =========================
# SERVE UPLOADED IMAGES
# =========================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
