import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # Statistical features
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.var(gray)
    ])

    # GLCM texture features
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    features.extend([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ])

    # Blur / edge feature
    features.append(np.mean(sobel(gray)))

    return np.array(features).reshape(1, -1)
