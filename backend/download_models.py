import os
from huggingface_hub import hf_hub_download

REPO_ID = "dikshant3697/biovision-models"
REPO_TYPE = "dataset"

FILES = [
    "blood_model.keras",
    "breast_image_model.keras",
    "breast_model.h5",
    "breast_scaler.pkl",
    "ecg_model.keras",
    "lung_model.h5",
    "router_model.keras",
    "skin_model.keras",
    "xray_model.keras",
]

TARGET_DIR = os.path.join("models", "weights")
os.makedirs(TARGET_DIR, exist_ok=True)

for file_name in FILES:
    print(f"Downloading {file_name}...")
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=file_name,
        repo_type=REPO_TYPE
    )

    target_path = os.path.join(TARGET_DIR, file_name)

    if not os.path.exists(target_path):
        with open(downloaded_path, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())

print("All models downloaded successfully.")