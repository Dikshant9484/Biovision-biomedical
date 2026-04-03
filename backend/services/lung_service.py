"""BioVision AI - Lung Cancer Detection Service"""
import numpy as np
import logging
from PIL import Image
from models.model_loader import load_image_model
from utils.image_utils import preprocess_image_256
from utils.feature_extractor import extract_image_features

logger = logging.getLogger(__name__)

LUNG_CLASSES = {0: 'Normal', 1: 'Malignant', 2: 'Benign'}

def predict_lung(img: Image.Image) -> dict:
    model, _ = load_image_model(
        'lung', 'LUNG_MODEL_PATH',
        'models/weights/lung_model.h5', num_classes=3
    )
    features = extract_image_features(img)

    if model is None:
        raise RuntimeError("Lung model is unavailable.")

    arr        = preprocess_image_256(img)
    preds      = model.predict(arr, verbose=0)[0]
    idx        = int(np.argmax(preds))
    label      = LUNG_CLASSES.get(idx, 'Unknown')
    confidence = float(preds[idx])

    return {
        "prediction":   label,
        "confidence":   round(confidence * 100, 2),
        "is_malignant": label == "Malignant",
        "risk_level":   "Low" if label == "Normal" else "High" if label == "Malignant" else "Moderate",
        "class_scores": {LUNG_CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(preds))},
        "features":     features,
        "model_type":   "ResNet50 Transfer Learning",
        "analysis_target": "Lung CT Scan"
    }
