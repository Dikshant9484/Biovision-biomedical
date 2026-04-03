"""BioVision AI - Universal Detector Route"""

from flask import Blueprint, request, jsonify
from utils.image_utils import validate_image_file, load_image_from_bytes
from services.universal_service import predict_universal
from services.chat_service import generate_medical_recommendation
import logging

logger = logging.getLogger(__name__)
universal_bp = Blueprint('universal', __name__)


@universal_bp.route('/universal', methods=['POST'])
def universal_predict():
    """POST /api/predict/universal - Auto-detect image type and classify"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']

        valid, error_msg = validate_image_file(file)
        if not valid:
            return jsonify({"error": error_msg}), 400

        img = load_image_from_bytes(file.read())
        result = predict_universal(img)

        context = (
            f"Universal Cancer Detector identified image as {result['category_display']} "
            f"(router confidence: {result['router_confidence']}%). "
            f"Specialist prediction: {result['prediction']} with {result['confidence']}% confidence."
        )

        result['ai_recommendation'] = generate_medical_recommendation(context)

        return jsonify(result)

    except Exception as e:
        logger.exception(f"Universal prediction error: {e}")
        return jsonify({"error": "Detection failed. Please try again."}), 500