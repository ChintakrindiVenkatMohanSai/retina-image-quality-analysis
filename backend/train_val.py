import os
import numpy as np
import joblib

from feature_extraction import extract_features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# PATHS (FIXED)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset folder outside backend
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))

# IMPORTANT: Save model INSIDE backend/artifacts
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

# --------------------------------------------------
# LOAD DATA FROM IMAGE FOLDERS
# --------------------------------------------------
def load_data(split):
    X, y = [], []
    split_dir = os.path.join(DATASET_DIR, split)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    print(f"\n📂 Scanning: {split_dir}")

    for cls in ["0", "1"]:   # 0 = bad, 1 = good
        folder = os.path.join(split_dir, cls)
        label = int(cls)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")

        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            try:
                features = extract_features(img_path)
                X.append(features.flatten())
                y.append(label)
            except Exception as e:
                print("⚠ Skipping:", img_path, e)

    return np.array(X), np.array(y)

# --------------------------------------------------
# TRAIN + VALIDATION DATA
# --------------------------------------------------
print("\n🚀 Loading training data...")
X_train, y_train = load_data("train")

print("\n🚀 Loading validation data...")
X_val, y_val = load_data("validation")

# --------------------------------------------------
# MODEL PIPELINE
# --------------------------------------------------
logreg = LogisticRegression(
    C=8.609404067363206,
    penalty="l2",
    solver="lbfgs",
    max_iter=100,
    random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rfe", RFE(estimator=logreg, n_features_to_select=15)),
    ("clf", logreg)
])

print("\n🧠 Training Logistic Regression (RFE-based)...")
pipeline.fit(X_train, y_train)

# --------------------------------------------------
# VALIDATION RESULTS
# --------------------------------------------------
val_pred = pipeline.predict(X_val)

print("\n📊 VALIDATION RESULTS")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_pred))

# --------------------------------------------------
# SAVE MODEL (FIXED PATH)
# --------------------------------------------------
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

model_path = os.path.join(ARTIFACTS_DIR, "logreg_rfe_best_by_test.pkl")
joblib.dump(pipeline, model_path)

print(f"\n✅ Model saved successfully at: {model_path}")
print("✅ Model type: Logistic Regression with RFE")