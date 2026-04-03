import numpy as np
import logging
from PIL import Image
from models.model_loader import load_image_model
from utils.image_utils import preprocess_image
from utils.feature_extractor import extract_image_features

logger = logging.getLogger(__name__)

def predict_skin(img: Image.Image) -> dict:
    """
    Classify a skin lesion dermoscopy image with a medical safety threshold.
    """
    model, is_mock = load_image_model(
        'skin',
        'SKIN_MODEL_PATH',
        'models/weights/skin_model.keras'
    )
    
    # 1. Extract visual features
    features = extract_image_features(img)
    features["lesion_border_irregularity"] = round(features["asymmetry_score"] / 10.0, 3)
    features["color_variation_score"] = round(features["std_intensity"] / 50.0, 3)
    features["dermoscopy_quality"] = "Acceptable" if features["contrast"] > 50 else "Low"
    
    if model is None:
        logger.error("Skin cancer model not found")
        raise RuntimeError("Skin cancer model is unavailable")

    try:
        arr = preprocess_image(img)
        # Raw prediction score (0.0 to 1.0)
        # 0 = Benign, 1 = Malignant
        raw_pred = model.predict(arr, verbose=0)[0][0]

        # --- MEDICAL SAFETY THRESHOLD ---
        # Instead of 0.5, we use 0.3. 
        # If the model is even 30% sure it's cancer, we flag it.
        THRESHOLD = 0.3 
        
        if raw_pred >= THRESHOLD:
            label = "Malignant"
            # Confidence is how much the model leans towards the Malignant class
            confidence_score = float(raw_pred)
        else:
            label = "Benign"
            # Confidence is how much it leans towards Benign
            confidence_score = float(1 - raw_pred)

    except Exception as e:
        logger.error(f"Skin cancer inference error: {e}")
        raise RuntimeError(f"Skin cancer prediction failed: {e}")
    
    return {
        "prediction": label,
        "confidence": round(confidence_score * 100, 2),
        "is_malignant": label == "Malignant",
        "lesion_type": "Melanoma (suspected)" if label == "Malignant" else "Benign lesion",
        "risk_level": _get_risk_level(confidence_score, label),
        "features": features,
        "abcde_flags": _get_abcde_flags(features),
        "model_type": "ResNet50 Transfer Learning (HAM10000)",
        "analysis_target": "Dermoscopy / Skin Lesion Image",
        "dataset_note": "Model trained on HAM10000 with Safety Thresholding"
    }

def _get_abcde_flags(features: dict) -> dict:
    return {
        "A_asymmetry": "Present" if features.get("asymmetry_score", 0) > 15 else "Absent",
        "B_border": "Irregular" if features.get("edge_density", 0) > 20 else "Regular",
        "C_color": "Variable" if features.get("std_intensity", 0) > 40 else "Uniform",
        "D_diameter": "Check with ruler",
        "E_evolution": "Monitor changes over time"
    }

def _get_risk_level(confidence: float, label: str) -> str:
    if label == "Benign":
        return "Low"
    # For Malignant, we categorize based on how high the score was
    if confidence < 0.50:
        return "Guarded/Follow-up"
    if confidence < 0.75:
        return "High Risk"
    return "Critical"