"""
BioVision AI - Breast Cancer Detection Service
Handles both image-based and tabular (risk estimator) breast cancer detection.
"""

import logging
import numpy as np
from PIL import Image

from models.model_loader import load_breast_tabular_model, load_image_model
from utils.image_utils import preprocess_image
from utils.feature_extractor import extract_image_features

logger = logging.getLogger(__name__)

BREAST_FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]


def predict_breast_image(img: Image.Image) -> dict:
    """
    Predict breast cancer from mammogram / histopathology image.

    Class mapping: {'benign': 0, 'malignant': 1}
        pred close to 0 = Benign
        pred close to 1 = Malignant
    Safety threshold = 0.3
    """
    try:
        if img is None:
            raise ValueError("No image provided for breast image prediction.")

        model, _ = load_image_model(
            'breast_image',
            'BREAST_IMAGE_MODEL_PATH',
            'models/weights/breast_image_model.keras'
        )

        if model is None:
            logger.error("Breast image model could not be loaded.")
            raise RuntimeError("Breast image model is unavailable.")

        features = _extract_breast_image_features(img)
        arr      = preprocess_image(img)
        pred     = float(model.predict(arr, verbose=0)[0][0])

        label, confidence_score = _interpret_binary_prediction(
            pred,
            positive_label="Malignant",
            negative_label="Benign"
        )

        return {
            "prediction":      label,
            "confidence":      round(confidence_score * 100, 2),
            "is_malignant":    label == "Malignant",
            "raw_score":       round(pred * 100, 2),
            "threshold_used":  0.3,
            "lesion_type":     "Malignant Tumor (suspected)" if label == "Malignant" else "Benign lesion",
            "risk_level":      _get_risk_level(confidence_score, label),
            "features":        features,
            "model_type":      "ResNet50V2 Transfer Learning (Breast Image)",
            "analysis_target": "Mammogram / Histopathology Image"
        }

    except Exception as e:
        logger.exception(f"Breast image prediction failed: {str(e)}")
        raise RuntimeError(f"Breast image prediction failed: {str(e)}")


def predict_breast_tabular(features: list) -> dict:
    """
    Predict breast cancer from 30 clinical biopsy features.

    Class mapping: {'benign': 0, 'malignant': 1}
    Safety threshold = 0.3
    """
    try:
        if features is None:
            raise ValueError("No clinical feature data provided.")

        if len(features) != 30:
            raise ValueError(f"Expected 30 features, got {len(features)}")

        model, scaler = load_breast_tabular_model()

        if model is None or scaler is None:
            logger.error("Breast tabular model or scaler could not be loaded.")
            raise RuntimeError("Breast tabular model is unavailable.")

        arr    = np.array(features, dtype=np.float32).reshape(1, -1)
        scaled = scaler.transform(arr)
        pred   = float(model.predict(scaled, verbose=0)[0][0])

        label, confidence_score = _interpret_binary_prediction(
            pred,
            positive_label="Malignant",
            negative_label="Benign"
        )

        feature_summary = {
            BREAST_FEATURE_NAMES[i]: round(float(features[i]), 4)
            for i in range(min(10, len(features)))
        }

        return {
            "prediction":              label,
            "confidence":              round(confidence_score * 100, 2),
            "is_malignant":            label == "Malignant",
            "raw_score":               round(pred * 100, 2),
            "threshold_used":          0.3,
            "risk_level":              _get_risk_level(confidence_score, label),
            "feature_summary":         feature_summary,
            "total_features_analyzed": 30,
            "model_type":              "Neural Network (30 Clinical Features)",
            "analysis_target":         "Clinical Biopsy Data"
        }

    except Exception as e:
        logger.exception(f"Breast tabular prediction failed: {str(e)}")
        raise RuntimeError(f"Breast tabular prediction failed: {str(e)}")


def _extract_breast_image_features(img: Image.Image) -> dict:
    """Extract visual features from breast medical images."""
    features = extract_image_features(img)
    features["tissue_density_estimate"]  = round(features.get("lesion_area_pct", 0) * 1.8, 2)
    features["edge_irregularity"]        = round(features.get("asymmetry_score", 0) / 10.0, 3)
    features["contrast_quality"]         = "Good" if features.get("contrast", 0) > 60 else "Poor"
    features["mass_indicator"]           = round(features.get("edge_density", 0) / 100.0, 3)
    return features


def _interpret_binary_prediction(
    pred: float,
    positive_label: str,
    negative_label: str
) -> tuple[str, float]:
    """
    Convert sigmoid output into label and confidence.

    Class mapping: {'benign': 0, 'malignant': 1}
        pred close to 0 = Benign
        pred close to 1 = Malignant

    Safety threshold 0.3:
        pred >= 0.3 → Malignant (flags suspicious cases early)
        pred <  0.3 → Benign
    """
    THRESHOLD = 0.3   # ← medical safety threshold

    if pred >= THRESHOLD:
        return positive_label, pred        # Malignant
    else:
        return negative_label, 1 - pred   # Benign


def _get_risk_level(confidence: float, label: str) -> str:
    """Determine risk level based on confidence and predicted label."""
    if label == "Benign":
        return "Low"
    if confidence < 0.50:
        return "Guarded / Follow-up"
    if confidence < 0.75:
        return "High Risk"
    return "Critical"