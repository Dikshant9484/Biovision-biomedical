"""
BioVision AI - Chest X-Ray (Pneumonia) Detection Service
Fixed: Conservative Thresholding + Review Zone + Honest Confidence
"""
import logging
from PIL import Image
from models.model_loader import load_image_model
from utils.image_utils import preprocess_image
from utils.feature_extractor import extract_image_features

logger = logging.getLogger(__name__)

def predict_xray(img: Image.Image) -> dict:
    """
    Predict pneumonia from chest X-ray with conservative medical screening logic.
    """

    try:
        # 1. Load model
        model, _ = load_image_model(
            'xray',
            'XRAY_MODEL_PATH',
            'models/weights/xray_model.keras'
        )

        if model is None:
            logger.error("Chest X-Ray model weights not found.")
            raise FileNotFoundError("X-Ray model weights missing at 'models/weights/xray_model.keras'")

        # 2. Extract image features
        features = extract_image_features(img)

        # 3. Preprocess and predict
        arr = preprocess_image(img)
        pred = float(model.predict(arr, verbose=0)[0][0])  # sigmoid output

        # 4. Conservative screening thresholds
        # --------------------------------------------------
        # < 0.35  -> Normal
        # 0.35-0.70 -> Needs Review
        # >= 0.70 -> Pneumonia Detected
        # --------------------------------------------------

        if pred < 0.35:
            label = "Normal"
            confidence = 1.0 - pred
            risk = "Low Risk"
            finding = "No strong pneumonia pattern detected"

        elif pred < 0.70:
            label = "Needs Review"
            confidence = pred
            risk = "Guarded / Follow-up"
            finding = "Possible subtle infiltrates or uncertain abnormality"

        else:
            label = "Pneumonia Detected"
            confidence = pred
            risk = "High Risk" if pred < 0.85 else "Critical Risk"
            finding = "Cloudy opacities / infiltrates suspected"

        return {
            "prediction": label,
            "confidence": round(min(confidence * 100, 99.2), 2),  # prevent fake 100%
            "is_pneumonia": label == "Pneumonia Detected",
            "risk_level": risk,
            "finding": finding,
            "raw_score": round(pred, 4),
            "features": features,
            "model_type": "ResNet50V2 (Chest X-Ray Pneumonia)",
            "analysis_target": "Chest X-Ray (AP/PA View)",
            "dataset_note": "Screening support only — model probability is not a diagnosis"
        }

    except Exception as e:
        logger.exception(f"X-Ray Prediction Error: {str(e)}")
        raise