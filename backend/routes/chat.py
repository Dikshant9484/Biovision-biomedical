"""BioVision AI - Groq Chatbot Route"""
from flask import Blueprint, request, jsonify
from services.chat_service import generate_medical_recommendation
import logging

logger = logging.getLogger(__name__)
chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/recommendation', methods=['POST'])
def chat_recommendation():
    """
    POST /api/chat/recommendation
    Body: { prediction_context, user_message, chat_history }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required"}), 400
        
        prediction_context = data.get('prediction_context', 'General medical inquiry')
        user_message = data.get('user_message', '')
        chat_history = data.get('chat_history', [])
        
        response = generate_medical_recommendation(
            prediction_context=prediction_context,
            user_message=user_message,
            chat_history=chat_history
        )
        
        return jsonify({
            "response": response,
            "disclaimer": "This AI system is for screening support only and is not a substitute for professional medical diagnosis."
        })
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "response": "I'm currently unable to process your request. Please consult a healthcare professional for medical guidance.",
            "error": str(e)
        }), 500
