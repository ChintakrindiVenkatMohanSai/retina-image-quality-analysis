import os
import joblib
import pickle

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from feature_extraction import extract_features


app = Flask(__name__)

# ============================================
# PATH SETUP
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================
# MODEL LOADING
# ============================================
MODEL_NAME = "Logistic Regression (RFE – 15 Quality Features)"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logreg_rfe_best_by_test.pkl")

model = None

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded using joblib")
except Exception:
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded using pickle")
    except Exception as e:
        print("❌ Model loading failed:", e)


# ============================================
# MAIN PAGE ROUTE
# ============================================
@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    confidence = None
    image_url = None

    if request.method == "POST":

        file = request.files.get("image")

        if file and file.filename and model:

            # Secure filename (important for security)
            filename = secure_filename(file.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            # Feature extraction
            features = extract_features(image_path)

            # Model prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0, 1] * 100

            result = (
                "GOOD QUALITY IMAGE"
                if prediction == 1
                else "BAD QUALITY RETINAL IMAGE"
            )

            confidence = round(probability, 2)

            # Image display path
            image_url = f"/uploads/{filename}"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_url=image_url,
        model_name=MODEL_NAME
    )


# ============================================
# SERVE UPLOADED IMAGES
# ============================================
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ============================================
# FEATURES DESCRIPTION PAGE
# ============================================
@app.route("/features")
def features():

    feature_descriptions = {

        "Mean":
        "Average brightness level of the retinal image. "
        "Higher values indicate brighter images.",

        "Std":
        "Standard deviation representing intensity variation "
        "and contrast strength.",

        "Skewness":
        "Measures asymmetry in pixel intensity distribution.",

        "Entropy":
        "Represents texture complexity. Higher entropy means "
        "more detailed retinal structures.",

        "Median":
        "Middle intensity value that reduces noise effects.",

        "Contrast_0":
        "Texture contrast measured horizontally (0° orientation).",

        "Correlation_0":
        "Spatial correlation between pixels at 0° direction.",

        "Homogeneity_0":
        "Texture smoothness in horizontal orientation.",

        "Contrast_45":
        "Diagonal texture contrast at 45° orientation.",

        "Energy_45":
        "Texture uniformity at 45° orientation.",

        "Homogeneity_45":
        "Smoothness of retinal texture diagonally.",

        "Energy_90":
        "Texture energy in vertical direction.",

        "Homogeneity_90":
        "Vertical structural uniformity.",

        "Contrast_135":
        "Anti-diagonal texture contrast.",

        "Homogeneity_135":
        "Smoothness across anti-diagonal structures."
    }

    return render_template("features.html", features=feature_descriptions)


# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    app.run(debug=True)
