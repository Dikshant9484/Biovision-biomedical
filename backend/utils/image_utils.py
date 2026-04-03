"""
BioVision AI - Image Utilities
Shared image preprocessing and validation helpers.
"""

import io
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Load PIL Image from raw bytes."""
    return Image.open(io.BytesIO(file_bytes)).convert('RGB')

def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """
    Preprocess a PIL image for ResNet50 inference.
    Returns a float32 numpy array of shape (1, H, W, 3).
    """
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

def preprocess_image_256(img: Image.Image) -> np.ndarray:
    """Preprocess for 256x256 models (lung)."""
    return preprocess_image(img, target_size=(256, 256))

def validate_image_file(file) -> tuple[bool, str]:
    """Validate an uploaded image file."""
    if not file:
        return False, "No file provided"
    if file.filename == '':
        return False, "No file selected"
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    return True, ""
