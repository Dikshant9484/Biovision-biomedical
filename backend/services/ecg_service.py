"""
BioVision AI - ECG Detection Service
ECG signal → 1D CNN classification pipeline (TensorFlow)
No PyTorch needed!

Dataset: MIT-BIH Arrhythmia Dataset (Kaggle)
Place CSV files at: datasets/ecg/
Expected CSV format: 188 columns (187 signal samples + 1 label column)

Pipeline:
  CSV ECG signal → preprocessing → 1D CNN → prediction
"""

import numpy as np
import logging
import io
import os

logger = logging.getLogger(__name__)


# ============================================================
# ECG Signal Processing
# ============================================================

def preprocess_ecg_signal(signal: np.ndarray, target_length: int = 187) -> np.ndarray:
    """Normalize and standardize ECG signal to fixed length."""
    mean = np.mean(signal)
    std  = np.std(signal) + 1e-8
    signal = (signal - mean) / std

    if len(signal) > target_length:
        indices = np.linspace(0, len(signal) - 1, target_length).astype(int)
        signal  = signal[indices]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')

    return signal.astype(np.float32)


# ============================================================
# Prediction
# ============================================================

def predict_ecg(signal_data: np.ndarray, fs: int = 360) -> dict:
    """
    Full ECG prediction pipeline:
      signal → preprocess → 1D CNN → prediction
    """
    from utils.feature_extractor import extract_ecg_features

    # Extract ECG features
    features = extract_ecg_features(signal_data, fs=fs)

    # Preprocess signal
    processed = preprocess_ecg_signal(signal_data)

    # Try loading CNN model
    model_path = os.getenv('ECG_MODEL_PATH', 'models/weights/ecg_model.keras')

    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            arr   = processed.reshape(1, 187, 1)
            pred  = model.predict(arr, verbose=0)[0][0]
            label = "Abnormal / Arrhythmia" if pred > 0.5 else "Normal"
            confidence = float(pred) if label != "Normal" else float(1 - pred)
            logger.info(f"✅ ECG CNN prediction: {label} ({confidence:.2f})")
        except Exception as e:
            logger.error(f"ECG CNN inference error: {e}")
            raise RuntimeError(f"ECG CNN inference failed: {e}")
    else:
        logger.error("ECG model not found")
        raise FileNotFoundError(f"ECG model not found at path: {model_path}")

    # Downsample signal for frontend chart
    display_signal = processed[::3].tolist()

    return {
        "prediction":     label,
        "confidence":     round(confidence * 100, 2),
        "is_abnormal":    label != "Normal",
        "risk_level":     "High" if (label != "Normal" and confidence > 0.75)
                          else "Moderate" if label != "Normal"
                          else "Low",
        "features":       features,
        "graph_stats": {
            "model_type":    "1D CNN (TensorFlow)",
            "signal_length": len(signal_data),
            "input_shape":   "187 x 1"
        },
        "signal_preview":  display_signal[:100],
        "model_type":      "1D Convolutional Neural Network",
        "analysis_target": "ECG Signal (CSV)",
        "pipeline":        "ECG CSV → Normalization → 1D CNN → Classification"
    }


def predict_ecg_from_csv_bytes(csv_bytes: bytes) -> dict:
    """Parse ECG CSV and run prediction."""
    import pandas as pd

    df = pd.read_csv(io.BytesIO(csv_bytes), header=None)

    if df.shape[1] >= 187:
        # MIT-BIH format: first row, first 187 columns = signal
        signal = df.iloc[0, :187].values.astype(np.float32)
    else:
        # Raw signal: entire first column
        signal = df.iloc[:, 0].values.astype(np.float32)

    return predict_ecg(signal)