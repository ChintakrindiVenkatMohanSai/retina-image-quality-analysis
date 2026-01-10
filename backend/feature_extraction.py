import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew
from collections import Counter
import math

def entropy_calc(pixels):
    counts = Counter(pixels)
    total = len(pixels)
    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pixels = gray.flatten()

    # ---------------- STATISTICAL ----------------
    mean = np.mean(pixels)
    std = np.std(pixels)
    skewness = skew(pixels)
    entropy = entropy_calc(pixels.tolist())
    median = np.median(pixels)

    # ---------------- GLCM ----------------
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    features = [
        mean,
        std,
        skewness,
        entropy,
        median,

        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],

        graycoprops(glcm, 'contrast')[0, 1],
        graycoprops(glcm, 'energy')[0, 1],
        graycoprops(glcm, 'homogeneity')[0, 1],

        graycoprops(glcm, 'energy')[0, 2],
        graycoprops(glcm, 'homogeneity')[0, 2],

        graycoprops(glcm, 'contrast')[0, 3],
        graycoprops(glcm, 'homogeneity')[0, 3],
    ]

    return np.array(features).reshape(1, -1)
