import os
import numpy as np
from feature_extraction import extract_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
ARTIFACTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts"))

model = joblib.load(os.path.join(ARTIFACTS_DIR, "quality_model.pkl"))
scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))

X_test, y_test = [], []

for label, cls in enumerate(["bad", "good"]):
    folder = os.path.join(DATASET_DIR, "test", cls)
    for img in os.listdir(folder):
        try:
            feat = extract_features(os.path.join(folder, img))
            X_test.append(feat.flatten())
            y_test.append(label)
        except:
            pass

X_test = scaler.transform(np.array(X_test))
y_test = np.array(y_test)

test_pred = model.predict(X_test)

print("\n📊 TEST RESULTS (FINAL)")
print("Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
