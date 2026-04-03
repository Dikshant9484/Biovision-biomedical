"""BioVision AI - Skin Cancer Routes"""
from flask import Blueprint, request, jsonify
from utils.image_utils import validate_image_file, load_image_from_bytes
from services.skin_service import predict_skin
from services.chat_service import generate_medical_recommendation
import logging

logger = logging.getLogger(__name__)
skin_bp = Blueprint('skin', __name__)

@skin_bp.route('/skin', methods=['POST'])
def skin_predict():
    """POST /api/predict/skin - Classify skin lesion dermoscopy image"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        valid, error_msg = validate_image_file(file)
        if not valid:
            return jsonify({"error": error_msg}), 400
        img = load_image_from_bytes(file.read())
        result = predict_skin(img)
        context = f"Skin cancer dermoscopy analysis predicted {result['prediction']} ({result.get('lesion_type', '')}) with {result['confidence']}% confidence."
        result['ai_recommendation'] = generate_medical_recommendation(context)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Skin prediction error: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500
