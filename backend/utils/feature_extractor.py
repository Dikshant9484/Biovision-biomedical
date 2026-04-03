"""
BioVision AI - Feature Extractor
Computes clinically meaningful image features from medical scans.
These features support the project's focus on "feature extraction" alongside classification.
"""

import numpy as np
from PIL import Image, ImageFilter
import logging

logger = logging.getLogger(__name__)


def extract_image_features(img: Image.Image) -> dict:
    """
    Extract technical image features from a medical image.
    Returns a dictionary of features with float values.
    """
    try:
        # Convert to grayscale for intensity analysis
        gray = img.convert('L')
        arr = np.array(gray, dtype=np.float32)
        
        # --- Intensity Statistics ---
        mean_intensity = float(np.mean(arr))
        std_intensity = float(np.std(arr))
        min_intensity = float(np.min(arr))
        max_intensity = float(np.max(arr))
        
        # Normalize to 0-1 range
        norm = arr / 255.0
        
        # --- Texture (GLCM approximation via local variance) ---
        # Simple texture measure: local standard deviation
        rows, cols = arr.shape
        texture_score = float(np.std(arr[::8, ::8]))  # Subsampled local variance
        
        # --- Edge Density (Sobel approximation) ---
        img_filtered = img.convert('L').filter(ImageFilter.FIND_EDGES)
        edge_arr = np.array(img_filtered, dtype=np.float32)
        edge_density = float(np.mean(edge_arr > 30))  # Proportion of edge pixels
        
        # --- Lesion Area (bright region approximation) ---
        threshold = mean_intensity + 0.5 * std_intensity
        lesion_mask = arr > threshold
        lesion_area_pct = float(np.sum(lesion_mask) / arr.size * 100)
        
        # --- Asymmetry Cue ---
        h, w = arr.shape
        left_half = arr[:, :w//2]
        right_half = np.fliplr(arr[:, w//2:])
        min_w = min(left_half.shape[1], right_half.shape[1])
        asymmetry_score = float(np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w])))
        
        # --- Contrast ---
        contrast = float(max_intensity - min_intensity)
        
        # --- Entropy approximation ---
        hist, _ = np.histogram(arr.flatten(), bins=64, range=(0, 256))
        hist_norm = hist / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))
        
        # --- Brightness uniformity ---
        uniformity = float(1.0 - (std_intensity / (mean_intensity + 1e-6)))
        
        return {
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "contrast": round(contrast, 2),
            "texture_score": round(texture_score, 2),
            "edge_density": round(edge_density * 100, 2),
            "lesion_area_pct": round(lesion_area_pct, 2),
            "asymmetry_score": round(asymmetry_score, 2),
            "entropy": round(entropy, 3),
            "brightness_uniformity": round(max(0, min(uniformity, 1.0)), 3),
            "image_width": img.width,
            "image_height": img.height
        }
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        return {
            "mean_intensity": 0,
            "std_intensity": 0,
            "contrast": 0,
            "texture_score": 0,
            "edge_density": 0,
            "lesion_area_pct": 0,
            "asymmetry_score": 0,
            "entropy": 0,
            "brightness_uniformity": 0,
            "image_width": 0,
            "image_height": 0
        }


def extract_ecg_features(signal: np.ndarray, fs: int = 360) -> dict:
    """
    Extract ECG-specific features from a 1D signal.
    Returns heart rate estimate, rhythm metrics, and signal statistics.
    """
    try:
        from scipy.signal import find_peaks
        
        # Normalize signal
        norm_signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        
        # --- Peak Detection (R-peaks) ---
        peaks, _ = find_peaks(norm_signal, height=0.5, distance=int(fs * 0.4))
        
        # --- Heart Rate ---
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / fs  # seconds
            heart_rate = round(60.0 / np.mean(rr_intervals), 1)
            # SDNN: std of RR intervals (rhythm irregularity)
            rhythm_irregularity = round(float(np.std(rr_intervals) * 1000), 2)  # ms
        else:
            heart_rate = 0.0
            rhythm_irregularity = 0.0
            rr_intervals = np.array([])
        
        # --- Signal Statistics ---
        signal_variance = round(float(np.var(signal)), 4)
        signal_mean = round(float(np.mean(signal)), 4)
        signal_rms = round(float(np.sqrt(np.mean(signal**2))), 4)
        
        # --- Frequency domain (simplified) ---
        fft = np.abs(np.fft.rfft(signal[:min(len(signal), 1024)]))
        dominant_freq_idx = np.argmax(fft[1:]) + 1
        freqs = np.fft.rfftfreq(min(len(signal), 1024), d=1.0/fs)
        dominant_freq = round(float(freqs[dominant_freq_idx]), 2) if dominant_freq_idx < len(freqs) else 0.0
        
        return {
            "heart_rate_bpm": heart_rate,
            "signal_variance": signal_variance,
            "signal_mean": signal_mean,
            "signal_rms": signal_rms,
            "peak_count": len(peaks),
            "rhythm_irregularity_ms": rhythm_irregularity,
            "dominant_frequency_hz": dominant_freq,
            "signal_length_samples": len(signal),
            "sampling_rate_hz": fs
        }
    except Exception as e:
        logger.error(f"ECG feature extraction error: {e}")
        return {
            "heart_rate_bpm": 0,
            "signal_variance": 0,
            "signal_mean": 0,
            "signal_rms": 0,
            "peak_count": 0,
            "rhythm_irregularity_ms": 0,
            "dominant_frequency_hz": 0,
            "signal_length_samples": 0,
            "sampling_rate_hz": 360
        }
