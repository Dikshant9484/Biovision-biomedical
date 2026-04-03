"""
BioVision AI — Dataset Organizer
Automatically organizes Kaggle dataset downloads into the required folder structure.

Usage:
    python organize_datasets.py --dataset skin   --source /path/to/downloaded/folder
    python organize_datasets.py --dataset blood  --source /path/to/downloaded/folder
    python organize_datasets.py --dataset breast --source /path/to/downloaded/folder
    python organize_datasets.py --dataset ecg    --source /path/to/downloaded/folder
    python organize_datasets.py --dataset router          # builds router from existing datasets
    python organize_datasets.py --check                   # check all dataset statuses
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

VAL_SPLIT = 0.15   # 15% of images go to validation set

def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def copy_images(src_dir, dst_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp')):
    """Copy all images from src_dir to dst_dir."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in src.rglob('*'):
        if f.suffix.lower() in extensions:
            shutil.copy2(f, dst / f.name)
            count += 1
    return count

def train_val_split(src_dir, train_dir, val_dir, val_ratio=VAL_SPLIT, extensions=('.jpg','.jpeg','.png','.bmp')):
    """Split images in src_dir into train/val folders."""
    src = Path(src_dir)
    files = [f for f in src.rglob('*') if f.suffix.lower() in extensions]
    random.shuffle(files)
    split = int(len(files) * (1 - val_ratio))
    train_files = files[:split]
    val_files = files[split:]

    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(val_dir).mkdir(parents=True, exist_ok=True)

    for f in train_files:
        shutil.copy2(f, Path(train_dir) / f.name)
    for f in val_files:
        shutil.copy2(f, Path(val_dir) / f.name)

    print(f"  → {len(train_files)} train / {len(val_files)} val")
    return len(train_files), len(val_files)


# ─── SKIN: HAM10000 ───────────────────────────────────────────────────────────
def organize_skin(source_dir):
    """
    Kaggle: kmader/skin-lesion-analysis-toward-melanoma-detection
    After download, you'll have:
        HAM10000_images_part_1/  and  HAM10000_images_part_2/
        HAM10000_metadata.csv

    Mapping:
        mel  (melanoma)        → malignant
        all others             → benign
    """
    print("\n🔬 Organizing Skin Cancer (HAM10000)...")
    import csv

    src = Path(source_dir)
    metadata_path = src / "HAM10000_metadata.csv"

    if not metadata_path.exists():
        # Try finding it recursively
        found = list(src.rglob("HAM10000_metadata.csv"))
        if not found:
            print("❌ HAM10000_metadata.csv not found. Download the full dataset from Kaggle.")
            return
        metadata_path = found[0]

    # Build image_id → file path map
    image_paths = {}
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        part_dir = src / part
        if not part_dir.exists():
            # try recursive find
            matches = list(src.rglob(f"{part}"))
            if matches:
                part_dir = matches[0]
        if part_dir.exists():
            for f in part_dir.glob("*.jpg"):
                image_paths[f.stem] = f

    if not image_paths:
        # fallback: scan all jpg files
        for f in src.rglob("*.jpg"):
            image_paths[f.stem] = f

    print(f"  Found {len(image_paths)} images")

    benign_train = "datasets/skin/train/benign"
    benign_val   = "datasets/skin/val/benign"
    mal_train    = "datasets/skin/train/malignant"
    mal_val      = "datasets/skin/val/malignant"
    make_dirs(benign_train, benign_val, mal_train, mal_val)

    benign_files = []
    malignant_files = []

    with open(metadata_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get('image_id', '').strip()
            dx = row.get('dx', '').strip().lower()
            if image_id in image_paths:
                if dx == 'mel':
                    malignant_files.append(image_paths[image_id])
                else:
                    benign_files.append(image_paths[image_id])

    print(f"  Benign: {len(benign_files)} | Malignant: {len(malignant_files)}")

    def split_and_copy(files, train_dir, val_dir):
        random.shuffle(files)
        split = int(len(files) * (1 - VAL_SPLIT))
        for f in files[:split]: shutil.copy2(f, Path(train_dir) / f.name)
        for f in files[split:]: shutil.copy2(f, Path(val_dir) / f.name)
        print(f"  → {split} train / {len(files)-split} val")

    split_and_copy(benign_files, benign_train, benign_val)
    split_and_copy(malignant_files, mal_train, mal_val)
    print("✅ Skin dataset organized → datasets/skin/")


# ─── BLOOD: ALL Leukemia ──────────────────────────────────────────────────────
def organize_blood(source_dir):
    """
    Kaggle: andrewmvd/leukemia-classification
    After download, you'll have folders like:
        C-NMC_Leukemia/
          training_data/fold_0/  (all/ hem/)
          testing_data/C-NMC_test_prelim_phase_data/

    OR Kaggle: nikhilsharma00/leukemia-dataset
        leukemia/  normal/  malignant/
    """
    print("\n🩸 Organizing Blood Cancer (Leukemia)...")
    src = Path(source_dir)

    normal_out_train   = "datasets/blood/train/normal"
    normal_out_val     = "datasets/blood/val/normal"
    leukemia_out_train = "datasets/blood/train/leukemia"
    leukemia_out_val   = "datasets/blood/val/leukemia"
    make_dirs(normal_out_train, normal_out_val, leukemia_out_train, leukemia_out_val)

    normal_imgs    = []
    leukemia_imgs  = []

    # Pattern 1: C-NMC dataset (all/ = leukemia, hem/ = healthy)
    for folder in src.rglob("all"):
        for f in folder.glob("*.bmp"): leukemia_imgs.append(f)
        for f in folder.glob("*.jpg"): leukemia_imgs.append(f)
    for folder in src.rglob("hem"):
        for f in folder.glob("*.bmp"): normal_imgs.append(f)
        for f in folder.glob("*.jpg"): normal_imgs.append(f)

    # Pattern 2: simple normal/ malignant/ folders
    for folder in src.rglob("normal"):
        if folder.is_dir():
            for f in folder.rglob("*"):
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    normal_imgs.append(f)
    for name in ["malignant", "leukemia", "cancer", "positive"]:
        for folder in src.rglob(name):
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                        leukemia_imgs.append(f)

    # Deduplicate
    normal_imgs   = list({f.name: f for f in normal_imgs}.values())
    leukemia_imgs = list({f.name: f for f in leukemia_imgs}.values())

    print(f"  Normal: {len(normal_imgs)} | Leukemia: {len(leukemia_imgs)}")

    def split_copy(files, train_dir, val_dir):
        random.shuffle(files)
        split = int(len(files) * (1 - VAL_SPLIT))
        for f in files[:split]: shutil.copy2(f, Path(train_dir) / f.name)
        for f in files[split:]: shutil.copy2(f, Path(val_dir) / f.name)
        print(f"  → {split} train / {len(files)-split} val")

    split_copy(normal_imgs,   normal_out_train,   normal_out_val)
    split_copy(leukemia_imgs, leukemia_out_train, leukemia_out_val)
    print("✅ Blood dataset organized → datasets/blood/")


# ─── BREAST IMAGE ─────────────────────────────────────────────────────────────
def organize_breast(source_dir):
    """
    Kaggle: ambarish/breakhis  OR  aryashah2k/breast-ultrasound-images-dataset
    BreakHis structure: benign/ malignant/ (nested by magnification and tumor type)
    Ultrasound: benign/ malignant/ normal/
    """
    print("\n🔬 Organizing Breast Image Dataset...")
    src = Path(source_dir)

    benign_train = "datasets/breast_image/train/benign"
    benign_val   = "datasets/breast_image/val/benign"
    mal_train    = "datasets/breast_image/train/malignant"
    mal_val      = "datasets/breast_image/val/malignant"
    make_dirs(benign_train, benign_val, mal_train, mal_val)

    benign_imgs   = []
    malignant_imgs = []

    for folder in src.rglob("benign"):
        if folder.is_dir():
            for f in folder.rglob("*"):
                if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp') and 'mask' not in f.name.lower():
                    benign_imgs.append(f)

    for name in ["malignant", "malignancy", "cancer"]:
        for folder in src.rglob(name):
            if folder.is_dir():
                for f in folder.rglob("*"):
                    if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp') and 'mask' not in f.name.lower():
                        malignant_imgs.append(f)

    benign_imgs   = list({f.name: f for f in benign_imgs}.values())
    malignant_imgs = list({f.name: f for f in malignant_imgs}.values())

    print(f"  Benign: {len(benign_imgs)} | Malignant: {len(malignant_imgs)}")

    def split_copy(files, train_dir, val_dir):
        random.shuffle(files)
        split = int(len(files) * (1 - VAL_SPLIT))
        for f in files[:split]: shutil.copy2(f, Path(train_dir) / f.name)
        for f in files[split:]: shutil.copy2(f, Path(val_dir) / f.name)
        print(f"  → {split} train / {len(files)-split} val")

    split_copy(benign_imgs,   benign_train, benign_val)
    split_copy(malignant_imgs, mal_train,   mal_val)
    print("✅ Breast image dataset organized → datasets/breast_image/")


# ─── ECG: MIT-BIH ─────────────────────────────────────────────────────────────
def organize_ecg(source_dir):
    """
    Kaggle: shayanfazeli/heartbeat
    After download: mitbih_train.csv  mitbih_test.csv
    """
    print("\n💓 Organizing ECG Dataset (MIT-BIH)...")
    src = Path(source_dir)

    os.makedirs("datasets/ecg", exist_ok=True)

    for fname in ["mitbih_train.csv", "mitbih_test.csv"]:
        matches = list(src.rglob(fname))
        if matches:
            shutil.copy2(matches[0], f"datasets/ecg/{fname}")
            print(f"  ✅ Copied {fname}")
        else:
            print(f"  ❌ {fname} not found in {source_dir}")

    print("✅ ECG dataset organized → datasets/ecg/")


# ─── ROUTER: build from existing datasets ─────────────────────────────────────
def organize_router(n_per_class=400):
    """
    Builds the router dataset by sampling from already-organized datasets.
    Requires skin, blood, and breast_image datasets to be set up first.
    Lung: samples from chest x-ray dataset if available.
    """
    print("\n🔀 Building Router Dataset from existing datasets...")

    SOURCES = {
        "breast": [
            "datasets/breast_image/train/benign",
            "datasets/breast_image/train/malignant",
        ],
        "skin": [
            "datasets/skin/train/benign",
            "datasets/skin/train/malignant",
        ],
        "blood": [
            "datasets/blood/train/normal",
            "datasets/blood/train/leukemia",
        ],
        "lung": [
            # Add your chest xray dataset path here if available
            # "datasets/chest_xray/train/NORMAL",
            # "datasets/chest_xray/train/PNEUMONIA",
        ],
    }

    for split in ["train", "val"]:
        for cls in SOURCES.keys():
            os.makedirs(f"datasets/router/{split}/{cls}", exist_ok=True)

    for cls, source_dirs in SOURCES.items():
        all_files = []
        for d in source_dirs:
            if os.path.exists(d):
                for f in Path(d).glob("*"):
                    if f.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'):
                        all_files.append(f)

        if not all_files:
            print(f"  ⚠️  No images found for class: {cls} — skipping")
            continue

        random.shuffle(all_files)
        sampled = all_files[:n_per_class]
        split_idx = int(len(sampled) * (1 - VAL_SPLIT))

        for f in sampled[:split_idx]:
            shutil.copy2(f, f"datasets/router/train/{cls}/{f.name}")
        for f in sampled[split_idx:]:
            shutil.copy2(f, f"datasets/router/val/{cls}/{f.name}")

        print(f"  {cls}: {split_idx} train / {len(sampled)-split_idx} val")

    print("✅ Router dataset built → datasets/router/")


# ─── CHECK ─────────────────────────────────────────────────────────────────────
def check_all():
    print("\n📊 Dataset Status Check\n")

    checks = {
        "Breast Tabular": (None, "Built-in sklearn — no download needed"),
        "Breast Image":   ("datasets/breast_image", ["train/benign", "train/malignant", "val/benign", "val/malignant"]),
        "Skin (HAM10000)": ("datasets/skin", ["train/benign", "train/malignant", "val/benign", "val/malignant"]),
        "Blood (Leukemia)": ("datasets/blood", ["train/normal", "train/leukemia", "val/normal", "val/leukemia"]),
        "ECG (MIT-BIH)":  ("datasets/ecg", ["mitbih_train.csv", "mitbih_test.csv"]),
        "Router":         ("datasets/router", ["train/breast", "train/skin", "train/blood", "val/breast"]),
    }

    weights = {
        "breast_model.h5":       "backend/models/weights/breast_model.h5",
        "breast_scaler.pkl":     "backend/models/weights/breast_scaler.pkl",
        "breast_image_model.h5": "backend/models/weights/breast_image_model.h5",
        "skin_model.h5":         "backend/models/weights/skin_model.h5",
        "blood_model.h5":        "backend/models/weights/blood_model.h5",
        "ecg_gcn.pt":            "backend/models/weights/ecg_gcn.pt",
        "router_model.h5":       "backend/models/weights/router_model.h5",
    }

    print("DATASETS:")
    for name, check in checks.items():
        if check[0] is None:
            print(f"  ✅  {name:<25} {check[1]}")
            continue
        base, subs = check
        if not os.path.exists(base):
            print(f"  ❌  {name:<25} Not found: {base}")
            continue
        missing = [s for s in subs if not os.path.exists(os.path.join(base, s))]
        if missing:
            print(f"  ⚠️   {name:<25} Missing: {missing}")
        else:
            # Count images
            count = sum(1 for _ in Path(base).rglob("*") if _.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.csv'))
            print(f"  ✅  {name:<25} Ready ({count} files)")

    print("\nMODEL WEIGHTS:")
    for name, path in weights.items():
        exists = os.path.exists(path)
        icon = "✅" if exists else "❌"
        size = f"({os.path.getsize(path)/1e6:.1f} MB)" if exists else "(not trained yet)"
        print(f"  {icon}  {name:<30} {size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioVision AI Dataset Organizer")
    parser.add_argument('--dataset', choices=['skin','blood','breast','ecg','router'], help='Which dataset to organize')
    parser.add_argument('--source', type=str, help='Path to downloaded Kaggle dataset folder')
    parser.add_argument('--check', action='store_true', help='Check status of all datasets and weights')
    args = parser.parse_args()

    if args.check:
        check_all()
    elif args.dataset == 'skin':
        if not args.source: print("❌ Provide --source path"); sys.exit(1)
        organize_skin(args.source)
    elif args.dataset == 'blood':
        if not args.source: print("❌ Provide --source path"); sys.exit(1)
        organize_blood(args.source)
    elif args.dataset == 'breast':
        if not args.source: print("❌ Provide --source path"); sys.exit(1)
        organize_breast(args.source)
    elif args.dataset == 'ecg':
        if not args.source: print("❌ Provide --source path"); sys.exit(1)
        organize_ecg(args.source)
    elif args.dataset == 'router':
        organize_router()
    else:
        parser.print_help()
