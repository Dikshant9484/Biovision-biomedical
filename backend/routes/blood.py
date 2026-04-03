"""BioVision AI - Blood Cancer Routes"""
from flask import Blueprint, request, jsonify
from utils.image_utils import validate_image_file, load_image_from_bytes
from services.blood_service import predict_blood
from services.chat_service import generate_medical_recommendation
import logging

logger = logging.getLogger(__name__)
blood_bp = Blueprint('blood', __name__)

@blood_bp.route('/blood', methods=['POST'])
def blood_predict():
    """POST /api/predict/blood - Classify blood smear microscopy image"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        valid, error_msg = validate_image_file(file)
        if not valid:
            return jsonify({"error": error_msg}), 400
        img = load_image_from_bytes(file.read())
        result = predict_blood(img)
        context = f"Blood cancer smear analysis predicted {result['prediction']} with {result['confidence']}% confidence."
        result['ai_recommendation'] = generate_medical_recommendation(context)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Blood prediction error: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500
