"""
BioVision AI - Universal Cancer Detector Service
Auto-detects image type and routes to the appropriate specialized classifier.
"""

import numpy as np
import logging
from PIL import Image

from models.model_loader import load_image_model
from utils.image_utils import preprocess_image, preprocess_image_256
from utils.feature_extractor import extract_image_features

from services import breast_service, skin_service, blood_service

logger = logging.getLogger(__name__)

# Router output classes (must match router training order)
ROUTER_CLASSES = ['blood', 'breast', 'lung', 'skin']

CATEGORY_INFO = {
    'breast': {
        'display': 'Breast Scan Image',
        'description': 'Mammogram or histopathology image detected',
        'icon': '🩺'
    },
    'skin': {
        'display': 'Skin Lesion Image',
        'description': 'Dermoscopy or skin lesion image detected',
        'icon': '🔬'
    },
    'blood': {
        'display': 'Blood Smear Image',
        'description': 'Blood smear microscopy image detected',
        'icon': '🩸'
    },
    'lung': {
        'display': 'Lung / Chest Scan',
        'description': 'Lung CT scan image detected',
        'icon': '🫁'
    },
    'unknown': {
        'display': 'Unknown Medical Image',
        'description': 'Image type could not be determined with confidence',
        'icon': '❓'
    }
}


def classify_image_type(img: Image.Image) -> tuple[str, float]:
    """
    Classify image type using router model.
    Returns: (category, confidence)
    """
    model, _ = load_image_model(
        'router',
        'ROUTER_MODEL_PATH',
        'models/weights/router_model.keras'
    )

    if model is None:
        logger.error("Router model could not be loaded.")
        raise RuntimeError("Router model is unavailable.")

    try:
        arr = preprocess_image(img)
        preds = model.predict(arr, verbose=0)[0]

        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])

        logger.info(f"[UNIVERSAL ROUTER] Raw preds: {preds}")
        logger.info(f"[UNIVERSAL ROUTER] Predicted class: {ROUTER_CLASSES[class_idx]}, Confidence: {confidence:.4f}")

        if confidence < 0.60:
            return "unknown", confidence

        return ROUTER_CLASSES[class_idx], confidence

    except Exception as e:
        logger.error(f"Router inference error: {e}")
        raise RuntimeError(f"Router prediction failed: {e}")


def predict_universal(img: Image.Image) -> dict:
    """
    Full universal detection pipeline:
    1. Detect image type using router model
    2. Route to correct specialist classifier
    3. Return combined result
    """
    detected_category, router_confidence = classify_image_type(img)
    category_info = CATEGORY_INFO.get(detected_category, CATEGORY_INFO['unknown'])

    try:
        if detected_category == 'breast':
            specialist_result = breast_service.predict_breast_image(img)

        elif detected_category == 'skin':
            specialist_result = skin_service.predict_skin(img)

        elif detected_category == 'blood':
            specialist_result = blood_service.predict_blood(img)

        elif detected_category == 'lung':
            specialist_result = _predict_lung(img)

        else:
            specialist_result = {
                "prediction": "Unclassified",
                "confidence": 0,
                "risk_level": "Unknown",
                "features": extract_image_features(img),
                "finding": "Image type could not be determined.",
                "note": "Please upload a clearer valid medical image."
            }

    except Exception as e:
        logger.error(f"Specialist model error: {e}")
        specialist_result = {
            "prediction": "Error",
            "confidence": 0,
            "risk_level": "Unknown",
            "features": extract_image_features(img),
            "finding": "Specialist model failed.",
            "note": str(e)
        }

    specialist_conf = float(specialist_result.get("confidence", 0))
    uncertainty_note = None
    if specialist_conf < 80:
        uncertainty_note = "Prediction confidence is moderate/low. Please verify using the dedicated specialist module."

    logger.info(f"[UNIVERSAL SPECIALIST] Category: {detected_category}")
    logger.info(f"[UNIVERSAL SPECIALIST] Result: {specialist_result}")

    final_response = {
        "category": detected_category,
        "detected_category": detected_category,
        "category_display": category_info['display'],
        "category_description": category_info['description'],
        "category_icon": category_info['icon'],
        "router_confidence": round(router_confidence * 100, 2),

        "prediction": specialist_result.get("prediction", "Unknown"),
        "confidence": specialist_result.get("confidence", 0),
        "risk_level": specialist_result.get("risk_level", "Unknown"),
        "finding": specialist_result.get("finding", ""),
        "features": specialist_result.get("features", {}),
        "specialist_result": specialist_result,

        "pipeline": f"Router → {category_info['display']} Classifier",
        "routing_note": "Image automatically classified and routed to the appropriate specialized AI model",
        "uncertainty_note": uncertainty_note
    }

    logger.info(f"[UNIVERSAL FINAL RESPONSE] {final_response}")
    return final_response


def _predict_lung(img: Image.Image) -> dict:
    """
    Lung cancer prediction using lung model
    """
    model, _ = load_image_model(
        'lung',
        'LUNG_MODEL_PATH',
        'models/weights/lung_model.h5'
    )

    features = extract_image_features(img)

    if model is None:
        raise RuntimeError("Lung model is unavailable.")

    try:
        arr = preprocess_image_256(img)
        preds = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(preds))

        # IMPORTANT: Make sure this label order matches your lung training dataset
        labels = ["Normal", "Malignant", "Benign"]
        label = labels[pred_idx]
        confidence = float(preds[pred_idx])

    except Exception as e:
        raise RuntimeError(f"Lung prediction failed: {e}")

    return {
        "prediction": label,
        "confidence": round(confidence * 100, 2),
        "is_malignant": label == "Malignant",
        "risk_level": "Low" if label == "Normal" else "High" if label == "Malignant" else "Moderate",
        "features": features,
        "model_type": "ResNet50 (Lung CT)",
        "finding": "Lung tissue abnormality detected" if label != "Normal" else "No strong abnormality detected"
    }