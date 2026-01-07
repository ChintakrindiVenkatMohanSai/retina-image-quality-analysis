import os
import numpy as np
from feature_extraction import extract_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts"))

def load_data(split):
    X, y = [], []
    split_dir = os.path.join(DATASET_DIR, split)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    print(f"\nScanning: {split_dir}")
    print("Found classes:", os.listdir(split_dir))

    for cls in ["0", "1"]:   # 0 = bad, 1 = good
        folder = os.path.join(split_dir, cls)
        label = int(cls)

        if not os.path.exists(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")

        for img in os.listdir(folder):
            img_path = os.path.join(folder, img)
            try:
                feat = extract_features(img_path)
                X.append(feat.flatten())
                y.append(label)
            except Exception as e:
                print("Skipping:", img_path, e)

    return np.array(X), np.array(y)

# ---------------- TRAIN ----------------
X_train, y_train = load_data("train")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------- VALIDATION ----------------
X_val, y_val = load_data("validation")
X_val = scaler.transform(X_val)

val_pred = model.predict(X_val)

print("\n📊 VALIDATION RESULTS")
print("Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))
print("Confusion Matrix:\n", confusion_matrix(y_val, val_pred))

# ---------------- SAVE MODEL ----------------
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
joblib.dump(model, os.path.join(ARTIFACTS_DIR, "quality_model.pkl"))
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

print("\n✅ Train + Validation completed. Model saved.")
