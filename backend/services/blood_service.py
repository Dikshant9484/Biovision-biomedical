"""
BioVision AI - Blood Cancer Detection Service
Blood smear / microscopic image classification for leukemia detection.

Dataset: Blood Cell Cancer (ALL) from Kaggle
Place dataset at: datasets/blood/
Structure:
  datasets/blood/train/normal/
  datasets/blood/train/leukemia/
  datasets/blood/val/normal/
  datasets/blood/val/leukemia/
"""

import logging
from PIL import Image

from models.model_loader import load_image_model
from utils.image_utils import preprocess_image
from utils.feature_extractor import extract_image_features

logger = logging.getLogger(__name__)


def predict_blood(img: Image.Image) -> dict:
    """
    Classify a blood smear microscopic image for leukemia/cancer detection.
    """
    try:
        if img is None:
            raise ValueError("No image provided for blood cancer prediction.")

        model, _ = load_image_model(
            'blood',
            'BLOOD_MODEL_PATH',
            'models/weights/blood_model.keras'
        )

        if model is None:
            logger.error("Blood cancer model could not be loaded.")
            raise RuntimeError("Blood cancer model is unavailable.")

        features = _extract_blood_features(img)
        arr      = preprocess_image(img)
        pred     = float(model.predict(arr, verbose=0)[0][0])

        label, confidence_score = _interpret_prediction(pred)

        return {
            "prediction":     label,
            "confidence":     round(confidence_score * 100, 2),
            "is_positive":    label != "Normal",
            "raw_score":      round(pred * 100, 2),
            "threshold_used": 0.7,
            "cancer_type": (
                "Acute Lymphoblastic Leukemia (suspected)"
                if label != "Normal"
                else None
            ),
            "risk_level":  _get_risk_level(confidence_score, label),
            "features":    features,
            "cell_analysis": {
                "abnormal_morphology": label != "Normal",
                "requires_pathology":  confidence_score > 0.60 and label != "Normal"
            },
            "model_type":      "ResNet50 Transfer Learning (Blood Smear)",
            "analysis_target": "Blood Smear Microscopic Image",
            "dataset_note":    "Model trained on ALL (Acute Lymphoblastic Leukemia) dataset from Kaggle"
        }

    except Exception as e:
        logger.exception(f"Blood cancer prediction failed: {str(e)}")
        raise RuntimeError(f"Blood cancer prediction failed: {str(e)}")


def _extract_blood_features(img: Image.Image) -> dict:
    """Extract generic image features and derive blood-smear-specific metrics."""
    features = extract_image_features(img)
    features["cell_density_estimate"] = round(features.get("lesion_area_pct", 0) * 2.3, 2)
    features["nuclear_irregularity"]  = round(features.get("asymmetry_score", 0) / 12.0, 3)
    features["stain_quality"]         = "Good" if features.get("contrast", 0) > 60 else "Poor"
    features["blast_cell_indicator"]  = round(features.get("edge_density", 0) / 100.0, 3)
    return features


def _interpret_prediction(pred: float) -> tuple[str, float]:
    """
    IMPORTANT — Class mapping from training:
        {'leukemia': 0, 'normal': 1}

    This means:
        pred close to 0.0 → Leukemia
        pred close to 1.0 → Normal

    Safety threshold = 0.7:
        if pred < 0.7 → flag as Leukemia (not confident enough it's normal)
        if pred >= 0.7 → Normal
    """
    THRESHOLD = 0.7   # ← flipped because leukemia=0, normal=1

    if pred < THRESHOLD:
        # Model leaning towards leukemia (class 0)
        confidence = 1 - pred   # higher when pred is closer to 0
        return "Blood Cancer / Leukemia", confidence
    else:
        # Model confident it is normal (class 1)
        return "Normal", pred


def _get_risk_level(confidence: float, label: str) -> str:
    """Determine risk level based on prediction confidence and label."""
    if label == "Normal":
        return "Low"
    if confidence < 0.50:
        return "Guarded / Follow-up"
    if confidence < 0.75:
        return "High Risk"
    return "Critical"
