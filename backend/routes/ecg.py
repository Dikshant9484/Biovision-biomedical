"""BioVision AI - ECG Detection Routes"""
from flask import Blueprint, request, jsonify
from services.ecg_service import predict_ecg_from_csv_bytes
from services.chat_service import generate_medical_recommendation
import logging

logger = logging.getLogger(__name__)
ecg_bp = Blueprint('ecg', __name__)

@ecg_bp.route('/ecg', methods=['POST'])
def ecg_predict():
    """POST /api/predict/ecg - Classify ECG signal from CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided. Upload an ECG CSV file."}), 400
        
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files are supported for ECG analysis"}), 400
        
        csv_bytes = file.read()
        result = predict_ecg_from_csv_bytes(csv_bytes)
        
        context = (
            f"ECG Graph Convolutional Network analysis predicted {result['prediction']} "
            f"with {result['confidence']}% confidence. "
            f"Estimated heart rate: {result['features'].get('heart_rate_bpm', 'N/A')} BPM. "
            f"Rhythm irregularity: {result['features'].get('rhythm_irregularity_ms', 'N/A')} ms."
        )
        result['ai_recommendation'] = generate_medical_recommendation(context)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"ECG prediction error: {e}")
        return jsonify({"error": f"ECG analysis failed: {str(e)}"}), 500
