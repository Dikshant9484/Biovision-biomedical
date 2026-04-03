import os
import requests

MODEL_DIR = "models/weights"
os.makedirs(MODEL_DIR, exist_ok=True)

FILES = {
    "blood_model.keras": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/blood_model.keras",
    "breast_image_model.keras": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/breast_image_model.keras",
    "breast_model.h5": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/breast_model.h5",
    "breast_scaler.pkl": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/breast_scaler.pkl",
    "ecg_model.keras": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/ecg_model.keras",
    "lung_model.h5": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/lung_model.h5",
    "router_model.keras": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/router_model.keras",
    "skin_model.keras": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/skin_model.keras",
    "xray_model.keras": "https://huggingface.co/datasets/dikshant3697/biovision-models/resolve/main/xray_model.keras",
}

def download_file(url, output_path):
    print(f"Downloading {output_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

for filename, url in FILES.items():
    output_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(output_path):
        print(f"Already exists: {filename}")
        continue

    download_file(url, output_path)

print("All models downloaded successfully.")