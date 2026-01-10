import os
import numpy as np
import joblib

from feature_extraction import extract_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts"))

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logreg_rfe_best_by_test.pkl")

# --------------------------------------------------
# LOAD MODEL (PIPELINE)
# --------------------------------------------------
model = joblib.load(MODEL_PATH)

print("✅ Loaded model:", type(model))

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------
X_test, y_test = [], []

test_dir = os.path.join(DATASET_DIR, "test")

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test folder not found: {test_dir}")

print(f"\n📂 Scanning test data: {test_dir}")

for cls in ["0", "1"]:  # 0 = bad, 1 = good
    folder = os.path.join(test_dir, cls)
    label = int(cls)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Missing class folder: {folder}")

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        try:
            features = extract_features(img_path)
            X_test.append(features.flatten())
            y_test.append(label)
        except Exception as e:
            print("⚠ Skipping:", img_path, e)

X_test = np.array(X_test)
y_test = np.array(y_test)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
print("\n📊 TEST RESULTS (FINAL)")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
