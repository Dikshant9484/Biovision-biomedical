"""
BioVision AI - Model Loader
Centralized model loading with lazy initialization and caching.
"""

import os
import logging
import tensorflow as tf
import joblib

logger = logging.getLogger(__name__)

# Global model registry (Memory Cache)
_models = {}

def load_breast_tabular_model():
    """Load the breast cancer tabular (risk estimator) model and its scaler."""
    if 'breast_tabular' not in _models:
        try:
            model_path  = os.getenv('BREAST_TABULAR_PATH', 'models/weights/breast_tabular_model.keras')
            scaler_path = os.getenv('BREAST_SCALER_PATH', 'models/weights/scaler.pkl')

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                _models['breast_tabular'] = tf.keras.models.load_model(model_path)
                _models['breast_scaler']  = joblib.load(scaler_path)
                logger.info("✅ Breast tabular model and scaler loaded")
            else:
                logger.warning(f"⚠️ Tabular files missing: {model_path} or {scaler_path}")
                _models['breast_tabular'] = None
                _models['breast_scaler']  = None
        except Exception as e:
            logger.error(f"❌ Error loading breast tabular: {e}")
            _models['breast_tabular'] = None

    return _models.get('breast_tabular'), _models.get('breast_scaler')


def load_image_model(model_key, env_var, default_path):
    """Generic loader for .keras image models."""
    if model_key not in _models:
        try:
            path = os.getenv(env_var, default_path)
            if os.path.exists(path):
                _models[model_key] = tf.keras.models.load_model(path)
                logger.info(f"✅ {model_key} model loaded from {path}")
            else:
                logger.error(f"⚠️ {model_key} weights not found at {path}")
                _models[model_key] = None
        except Exception as e:
            logger.error(f"❌ Failed to load {model_key}: {e}")
            _models[model_key] = None

    # Return model and a 'is_multiclass' flag (False for binary models)
    return _models.get(model_key), False


def preload_all_models():
    """Call this on server startup to warm up the cache."""
    logger.info("🚀 Preloading all medical models into memory...")
    
    # 1. Breast Models
    load_breast_tabular_model()
    load_image_model('breast_image', 'BREAST_IMG_PATH', 'models/weights/breast_image_model.keras')
    
    # 2. Skin Cancer
    load_image_model('skin', 'SKIN_MODEL_PATH', 'models/weights/skin_model.keras')
    
    # 3. Blood Cancer (Leukemia)
    load_image_model('blood', 'BLOOD_MODEL_PATH', 'models/weights/blood_model.keras')
    
    # 4. Chest X-Ray (Pneumonia)
    load_image_model('xray', 'XRAY_MODEL_PATH', 'models/weights/xray_model.keras')
    
    # 5. ECG (Heart Disease) - Future model
    load_image_model('ecg', 'ECG_MODEL_PATH', 'models/weights/ecg_model.keras')

    logger.info("🎯 All available models are now in memory.")