"""
BioVision AI — Master Training Script
Run this to train ALL models in sequence.

Usage:
    python train_all.py                  # train everything
    python train_all.py --skip-ecg       # skip ECG (needs PyTorch Geometric)
    python train_all.py --only breast    # train one model only

Order:
    1. Breast Tabular (no dataset needed - uses sklearn)
    2. Breast Image   (needs: datasets/breast_image/)
    3. Skin           (needs: datasets/skin/)
    4. Blood          (needs: datasets/blood/)
    5. ECG GCN        (needs: datasets/ecg/)
    6. Router         (needs: datasets/router/)
"""

import os
import sys
import time
import argparse

def header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_dataset(path, required_subdirs):
    """Return True if dataset folder exists with required structure."""
    if not os.path.exists(path):
        return False, f"Folder not found: {path}"
    for sub in required_subdirs:
        if not os.path.exists(os.path.join(path, sub)):
            return False, f"Missing subfolder: {os.path.join(path, sub)}"
    return True, "OK"

def run_training(script_path, label):
    header(f"Training: {label}")
    start = time.time()
    ret = os.system(f"python {script_path}")
    elapsed = time.time() - start
    if ret == 0:
        print(f"\n✅ {label} — done in {elapsed/60:.1f} min")
        return True
    else:
        print(f"\n❌ {label} — FAILED (exit code {ret})")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', type=str, default=None,
                        help='Train only one model: breast_tabular | breast_image | skin | blood | ecg | router')
    parser.add_argument('--skip-ecg', action='store_true', help='Skip ECG GCN training')
    args = parser.parse_args()

    os.makedirs("backend/models/weights", exist_ok=True)
    results = {}

    SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "training")

    TASKS = [
        {
            "id": "breast_tabular",
            "label": "Breast Tabular MLP",
            "script": os.path.join(SCRIPTS_DIR, "breast_train.py"),
            "dataset_check": None,   # uses sklearn built-in
        },
        {
            "id": "breast_image",
            "label": "Breast Image ResNet50",
            "script": os.path.join(SCRIPTS_DIR, "breast_image_train.py"),
            "dataset_check": ("datasets/breast_image", ["train/benign", "train/malignant", "val/benign", "val/malignant"]),
        },
        {
            "id": "skin",
            "label": "Skin Cancer ResNet50 (HAM10000)",
            "script": os.path.join(SCRIPTS_DIR, "skin_train.py"),
            "dataset_check": ("datasets/skin", ["train/benign", "train/malignant", "val/benign", "val/malignant"]),
        },
        {
            "id": "blood",
            "label": "Blood Cancer ResNet50 (Leukemia)",
            "script": os.path.join(SCRIPTS_DIR, "blood_train.py"),
            "dataset_check": ("datasets/blood", ["train/normal", "train/leukemia", "val/normal", "val/leukemia"]),
        },
        {
            "id": "ecg",
            "label": "ECG Graph Convolutional Network",
            "script": os.path.join(SCRIPTS_DIR, "ecg_train.py"),
            "dataset_check": ("datasets/ecg", ["mitbih_train.csv", "mitbih_test.csv"]),
        },
        {
            "id": "router",
            "label": "Universal Router Classifier",
            "script": os.path.join(SCRIPTS_DIR, "router_train.py"),
            "dataset_check": ("datasets/router", ["train/breast", "train/skin", "train/blood", "train/lung", "val/breast"]),
        },
    ]

    for task in TASKS:
        if args.only and task["id"] != args.only:
            continue
        if args.skip_ecg and task["id"] == "ecg":
            print(f"\n⏭  Skipping ECG (--skip-ecg)")
            continue

        # Check dataset
        if task["dataset_check"]:
            path, subdirs = task["dataset_check"]
            ok, msg = check_dataset(path, subdirs)
            if not ok:
                print(f"\n⚠️  SKIPPING {task['label']}")
                print(f"   Dataset not ready: {msg}")
                print(f"   See TRAINING_GUIDE.md for download instructions")
                results[task["id"]] = "SKIPPED"
                continue

        ok = run_training(task["script"], task["label"])
        results[task["id"]] = "OK" if ok else "FAILED"

    # Summary
    header("Training Summary")
    for k, v in results.items():
        icon = "✅" if v == "OK" else ("⏭" if v == "SKIPPED" else "❌")
        print(f"  {icon}  {k:<20} {v}")

if __name__ == "__main__":
    main()
