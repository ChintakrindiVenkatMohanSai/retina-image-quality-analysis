import os
import joblib
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # IMPORTANT for Flask plotting
import matplotlib.pyplot as plt
import time

from flask import Flask, render_template, request, send_from_directory
from feature_extraction import extract_features

app = Flask(__name__)

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
MODEL_NAME = "Logistic Regression (RFE – 15 Quality Features)"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logreg_rfe_best_by_test.pkl")

print("Loading model from:", MODEL_PATH)

model = None

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded using joblib.")
except Exception as e:
    print("Joblib load failed:", e)
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Model loaded using pickle.")
    except Exception as e2:
        print("Model loading failed completely:", e2)

# =========================
# FEATURE NAMES
# =========================
FEATURE_NAMES = [
    "Mean", "Std", "Skewness", "Entropy", "Median",
    "Contrast_0", "Correlation_0", "Homogeneity_0",
    "Contrast_45", "Energy_45", "Homogeneity_45",
    "Energy_90", "Homogeneity_90",
    "Contrast_135", "Homogeneity_135"
]

# =========================
# EXPLAINABILITY FUNCTION
# =========================
def generate_explainability_plot(model, features):
    try:
        scaler = model.named_steps["scaler"]
        rfe = model.named_steps["rfe"]
        clf = model.named_steps["clf"]

        # Apply same preprocessing
        X_scaled = scaler.transform(features)
        X_selected = rfe.transform(X_scaled)

        # Selected feature names
        selected_features = [
            f for f, s in zip(FEATURE_NAMES, rfe.support_) if s
        ]

        # Contribution calculation
        coef = clf.coef_[0]
        contributions = X_selected.flatten() * coef

        # Plot
        plt.figure(figsize=(12,6))
        plt.barh(selected_features, contributions)
        plt.xlabel("Feature Contribution")
        plt.title("Retinal Image Explainability")

        filename = f"explain_{int(time.time())}.png"
        path = os.path.join(STATIC_DIR, filename)

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Explainability plot saved:", path)
        return filename

    except Exception as e:
        print("Explainability Error:", e)
        return None


# =========================
# MAIN ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_url = None
    extracted_features = None
    shap_image = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "" and model is not None:
            filename = file.filename
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            # Feature extraction
            features = extract_features(image_path)

            # Prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0, 1] * 100

            result = "GOOD QUALITY IMAGE" if prediction == 1 else "BAD QUALITY IMAGE"
            confidence = round(probability, 2)

            # Explainability (IMPORTANT LINE)
            shap_image = generate_explainability_plot(model, features)

            image_url = f"/uploads/{filename}"
            extracted_features = FEATURE_NAMES

        elif model is None:
            result = "MODEL NOT LOADED — CHECK MODEL FILE"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        extracted_features=extracted_features,
        model_name=MODEL_NAME,
        shap_image=shap_image
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