"""
BioVision AI - Flask Routes
Health check endpoint
"""
from flask import Blueprint, jsonify
import os

health_bp = Blueprint('health', __name__)

@health_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "app": "BioVision AI",
        "version": "1.0.0",
        "groq_configured": bool(os.getenv('GROQ_API_KEY')),
        "models_dir": os.path.exists('models/weights')
    })
