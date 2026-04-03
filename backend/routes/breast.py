"""BioVision AI - Breast Cancer Routes"""
from flask import Blueprint, request, jsonify
from utils.image_utils import validate_image_file, load_image_from_bytes
from services.breast_service import predict_breast_image, predict_breast_tabular
from services.chat_service import generate_medical_recommendation
import logging

logger = logging.getLogger(__name__)
breast_bp = Blueprint('breast', __name__)


@breast_bp.route('/breast-image', methods=['POST'])
def breast_image_predict():
    """POST /api/predict/breast-image - Classify mammogram/histopathology image"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        valid, error_msg = validate_image_file(file)
        if not valid:
            return jsonify({"error": error_msg}), 400
        
        img = load_image_from_bytes(file.read())
        result = predict_breast_image(img)
        
        # Auto-generate AI recommendation
        context = f"Breast cancer image analysis predicted {result['prediction']} with {result['confidence']}% confidence."
        result['ai_recommendation'] = generate_medical_recommendation(context)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Breast image prediction error: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


@breast_bp.route('/breast-tabular', methods=['POST'])
def breast_tabular_predict():
    """POST /api/predict/breast-tabular - Classify from 30 clinical features"""
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "Missing 'features' array in request body"}), 400
        
        features = data['features']
        if len(features) != 30:
            return jsonify({"error": f"Expected 30 features, got {len(features)}"}), 400
        
        result = predict_breast_tabular(features)
        
        context = f"Breast cancer clinical risk estimator predicted {result['prediction']} with {result['confidence']}% confidence based on 30 biopsy features."
        result['ai_recommendation'] = generate_medical_recommendation(context)
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Breast tabular prediction error: {e}")
        return jsonify({"error": "Prediction failed. Please try again."}), 500
