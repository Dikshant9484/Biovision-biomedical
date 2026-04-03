"""BioVision AI - Chest X-Ray Route"""
from flask import Blueprint, request, jsonify
from utils.image_utils import validate_image_file, load_image_from_bytes
from services.xray_service import predict_xray
from services.chat_service import generate_medical_recommendation
import logging

logger  = logging.getLogger(__name__)
xray_bp = Blueprint('xray', __name__)

@xray_bp.route('/xray', methods=['POST'])
def xray_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        valid, msg = validate_image_file(file)
        if not valid:
            return jsonify({"error": msg}), 400
        img    = load_image_from_bytes(file.read())
        result = predict_xray(img)
        ctx    = f"Chest X-Ray: {result['prediction']} with {result['confidence']}% confidence. {result.get('finding','')}"
        result['ai_recommendation'] = generate_medical_recommendation(ctx)
        return jsonify(result)
    except Exception as e:
        logger.error(f"X-Ray error: {e}")
        return jsonify({"error": str(e)}), 500