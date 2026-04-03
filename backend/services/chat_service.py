"""
BioVision AI - Groq Chatbot Service
Context-aware medical AI assistant using Groq's LLM API.
IMPORTANT: This is a screening support tool - never provides final diagnosis.
"""

import os
import logging
from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are BioVision AI Assistant, a helpful medical screening support system.

IMPORTANT RULES:
1. You NEVER provide a final medical diagnosis
2. You ALWAYS include a disclaimer that this is screening support only
3. You provide professional, empathetic, medically-informed suggestions
4. You recommend consulting healthcare professionals for all findings
5. You explain results in clear, understandable language
6. You stay focused on the provided prediction context

Your responses should be:
- Compassionate and professional
- 3-5 bullet points of actionable next steps
- Clear about the AI's limitations
- Encouraging toward professional medical consultation

Always end with: "⚠️ This AI system is for screening support only and is not a substitute for professional medical diagnosis. Please consult a qualified healthcare provider."
"""

def get_groq_client() -> Groq | None:
    """Initialize Groq client with API key from environment."""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        logger.warning("GROQ_API_KEY not set - chatbot will use fallback responses")
        return None
    return Groq(api_key=api_key)


def generate_medical_recommendation(
    prediction_context: str,
    user_message: str = None,
    chat_history: list = None
) -> str:
    """
    Generate a context-aware medical recommendation using Groq.
    
    Args:
        prediction_context: String describing the model's prediction result
        user_message: Optional follow-up question from the user
        chat_history: Optional conversation history for multi-turn chat
    
    Returns:
        AI-generated medical recommendation string
    """
    client = get_groq_client()
    
    if not client:
        return _fallback_recommendation(prediction_context)
    
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-6:]:  # Keep last 6 messages for context
                messages.append(msg)
        
        # Build the user prompt
        if user_message:
            prompt = f"""Medical Screening Context: {prediction_context}

User Question: {user_message}

Please respond to the user's question in the context of the above screening result."""
        else:
            prompt = f"""Medical Screening Result: {prediction_context}

Please provide professional next-step recommendations and medical guidance based on this result."""
        
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            max_tokens=600,
            temperature=0.4
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        return _fallback_recommendation(prediction_context)


def _fallback_recommendation(context: str) -> str:
    """Fallback response when Groq API is unavailable."""
    context_lower = context.lower()
    
    if 'malignant' in context_lower or 'cancer' in context_lower or 'abnormal' in context_lower:
        return """Based on the screening result, here are recommended next steps:

• **Immediate Consultation**: Schedule an appointment with a specialist (oncologist/radiologist) as soon as possible
• **Do Not Delay**: Early detection and professional evaluation are critical for the best outcomes
• **Gather Records**: Compile all medical history, previous scans, and test results for your doctor
• **Second Opinion**: Consider seeking a second opinion from a qualified specialist
• **Lifestyle**: Maintain a healthy diet and avoid smoking or excessive alcohol consumption

⚠️ This AI system is for screening support only and is not a substitute for professional medical diagnosis. Please consult a qualified healthcare provider."""
    
    elif 'benign' in context_lower or 'normal' in context_lower:
        return """Based on the screening result, here are recommended next steps:

• **Regular Monitoring**: Continue routine check-ups and follow-up screenings as recommended by your doctor
• **Stay Proactive**: Report any new symptoms or changes to your healthcare provider promptly
• **Healthy Lifestyle**: Maintain a balanced diet, regular exercise, and avoid risk factors
• **Annual Screenings**: Keep up with age-appropriate cancer screenings and medical check-ups
• **Peace of Mind**: While results appear normal, always follow up with a healthcare professional

⚠️ This AI system is for screening support only and is not a substitute for professional medical diagnosis. Please consult a qualified healthcare provider."""
    
    else:
        return """Here are general health recommendations:

• **Consult a Specialist**: Discuss the screening results with a qualified healthcare provider
• **Follow-up Testing**: Additional diagnostic tests may be recommended based on your results
• **Document Symptoms**: Keep a record of any symptoms or changes you experience
• **Stay Informed**: Ask your doctor to explain the results and what they mean for your health
• **Support Network**: Consider connecting with patient support groups if needed

⚠️ This AI system is for screening support only and is not a substitute for professional medical diagnosis. Please consult a qualified healthcare provider."""
