# BioVision AI — Complete Model Training Guide

Follow this guide **in order** to train all 6 models from scratch.
Total estimated time: 3–5 hours depending on your GPU.

---

## Prerequisites

### Install Python dependencies
```bash
cd biovision-complete/backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Install Kaggle CLI (for easy dataset download)
```bash
pip install kaggle

# Set up your Kaggle API key:
# 1. Go to https://www.kaggle.com → Account → Create New Token
# 2. Download kaggle.json
# 3. Place it at:
#    Linux/Mac: ~/.kaggle/kaggle.json
#    Windows:   C:\Users\<user>\.kaggle\kaggle.json
```

### Check everything is ready
```bash
# From the biovision-complete/ root folder:
python organize_datasets.py --check
```

---

## MODEL 1 — Breast Tabular MLP
**Time: ~1 minute | No dataset download needed**

This uses the built-in Wisconsin Breast Cancer dataset from scikit-learn.

```bash
cd biovision-complete
python training/breast_train.py
```

**Output:**
```
backend/models/weights/breast_model.h5
backend/models/weights/breast_scaler.pkl
```

**OR** if you already have these from your old project:
```bash
cp old_project/breast_model.h5   backend/models/weights/breast_model.h5
cp old_project/breast_scaler.pkl backend/models/weights/breast_scaler.pkl
```

---

## MODEL 2 — Breast Image (ResNet50)
**Time: 30–60 min | Dataset: ~3 GB**

### Step 1 — Download dataset
```bash
# Option A: BreakHis (histopathology) — RECOMMENDED
kaggle datasets download -d ambarish/breakhis
unzip breakhis.zip -d downloads/breakhis

# Option B: Breast Ultrasound Images
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset
unzip breast-ultrasound-images-dataset.zip -d downloads/breast_ultrasound
```

### Step 2 — Organize dataset
```bash
python organize_datasets.py --dataset breast --source downloads/breakhis
# OR
python organize_datasets.py --dataset breast --source downloads/breast_ultrasound
```

Verify:
```
datasets/breast_image/
  train/benign/      ← ~1400 images
  train/malignant/   ← ~2000 images
  val/benign/        ← ~250 images
  val/malignant/     ← ~350 images
```

### Step 3 — Train
```bash
python training/breast_image_train.py
```

**Output:** `backend/models/weights/breast_image_model.h5`

---

## MODEL 3 — Skin Cancer (ResNet50 + HAM10000)
**Time: 30–60 min | Dataset: ~3 GB**

### Step 1 — Download dataset
```bash
kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection
unzip skin-lesion-analysis-toward-melanoma-detection.zip -d downloads/ham10000
```

You should see these files:
```
downloads/ham10000/
  HAM10000_images_part_1/    (5000 images)
  HAM10000_images_part_2/    (5000 images)
  HAM10000_metadata.csv
```

### Step 2 — Organize dataset
```bash
python organize_datasets.py --dataset skin --source downloads/ham10000
```

The script reads HAM10000_metadata.csv and maps:
- `mel` (melanoma) → malignant
- `nv`, `bkl`, `df`, `akiec`, `bcc`, `vasc` → benign

Verify:
```
datasets/skin/
  train/benign/      ← ~7000 images
  train/malignant/   ← ~850 images
  val/benign/        ← ~1200 images
  val/malignant/     ← ~150 images
```

### Step 3 — Train
```bash
python training/skin_train.py
```

**Output:** `backend/models/weights/skin_model.h5`

> **Note:** HAM10000 is class-imbalanced (more benign than malignant).
> The training script handles this with augmentation.

---

## MODEL 4 — Blood Cancer / Leukemia (ResNet50)
**Time: 20–40 min | Dataset: ~1 GB**

### Step 1 — Download dataset
```bash
# Option A: C-NMC Leukemia (recommended, better quality)
kaggle datasets download -d andrewmvd/leukemia-classification
unzip leukemia-classification.zip -d downloads/leukemia

# Option B: Simpler leukemia dataset
kaggle datasets download -d nikhilsharma00/leukemia-dataset
unzip leukemia-dataset.zip -d downloads/leukemia
```

### Step 2 — Organize dataset
```bash
python organize_datasets.py --dataset blood --source downloads/leukemia
```

Verify:
```
datasets/blood/
  train/normal/     ← ~1500 images
  train/leukemia/   ← ~2000 images
  val/normal/       ← ~260 images
  val/leukemia/     ← ~350 images
```

### Step 3 — Train
```bash
python training/blood_train.py
```

**Output:** `backend/models/weights/blood_model.h5`

---

## MODEL 5 — ECG Graph Convolutional Network
**Time: 20–40 min | Dataset: ~170 MB**

This model is different — it uses PyTorch Geometric, not TensorFlow.

### Step 1 — Install PyTorch + PyTorch Geometric
```bash
# Install PyTorch (CPU version — works on any machine)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
pip install torch-geometric

# Install additional dependencies
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
```

> If you have a GPU, use the GPU version of PyTorch for faster training.
> See: https://pytorch.org/get-started/locally/

### Step 2 — Download dataset
```bash
kaggle datasets download -d shayanfazeli/heartbeat
unzip heartbeat.zip -d downloads/ecg
```

You should see:
```
downloads/ecg/
  mitbih_train.csv    (87,554 rows × 188 columns)
  mitbih_test.csv     (21,892 rows × 188 columns)
```

**CSV format:** 187 signal values + 1 label column (0=Normal, 1-4=Arrhythmia types)

### Step 3 — Organize dataset
```bash
python organize_datasets.py --dataset ecg --source downloads/ecg
```

Verify:
```
datasets/ecg/
  mitbih_train.csv
  mitbih_test.csv
```

### Step 4 — Train
```bash
python training/ecg_train.py
```

**Output:** `backend/models/weights/ecg_gcn.pt`

---

## MODEL 6 — Universal Router Classifier
**Time: 15–30 min | No new dataset download**

This model learns to distinguish between image types (breast/skin/blood/lung).
It uses images sampled from the datasets you already organized above.

**Requirements:** Models 2, 3, and 4 datasets must be organized first.

### Step 1 — Build router dataset
```bash
python organize_datasets.py --dataset router
```

This samples 400 images per class from existing datasets:
```
datasets/router/
  train/breast/     ← 340 images
  train/skin/       ← 340 images
  train/blood/      ← 340 images
  train/lung/       ← 340 images (if chest x-ray available)
  val/breast/       ← 60 images
  val/skin/         ← 60 images
  ...
```

### Step 2 — Train
```bash
python training/router_train.py
```

**Output:** `backend/models/weights/router_model.h5`

---

## Train All At Once

After setting up all datasets, you can train everything in one command:

```bash
python train_all.py
```

Or skip the ECG model (needs PyTorch Geometric):
```bash
python train_all.py --skip-ecg
```

Or train just one model:
```bash
python train_all.py --only skin
```

---

## Verify All Models Are Ready

```bash
python organize_datasets.py --check
```

Expected output:
```
DATASETS:
  ✅  Breast Tabular            Built-in sklearn — no download needed
  ✅  Breast Image              Ready (3600 files)
  ✅  Skin (HAM10000)           Ready (9000 files)
  ✅  Blood (Leukemia)          Ready (4000 files)
  ✅  ECG (MIT-BIH)             Ready (2 files)
  ✅  Router                    Ready (1600 files)

MODEL WEIGHTS:
  ✅  breast_model.h5           (0.3 MB)
  ✅  breast_scaler.pkl         (0.01 MB)
  ✅  breast_image_model.h5     (94.5 MB)
  ✅  skin_model.h5             (94.5 MB)
  ✅  blood_model.h5            (94.5 MB)
  ✅  ecg_gcn.pt                (2.1 MB)
  ✅  router_model.h5           (94.5 MB)
```

---

## Run the App

Once all models are trained:

```bash
# Backend
cd backend
python app.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — all models are now live!

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch_geometric` | Run: `pip install torch-geometric` |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in training script |
| Training very slow | Add GPU or reduce `EPOCHS` |
| `ValueError: No images found` | Run `--check` to verify dataset structure |
| Skin model low accuracy | HAM10000 is imbalanced — normal, increase epochs |
| `kaggle: command not found` | Run: `pip install kaggle` and set up kaggle.json |

---

## Dataset Summary

| Model | Kaggle Dataset | Size | Approx Images |
|-------|---------------|------|---------------|
| Breast Image | ambarish/breakhis | 3 GB | 7,909 |
| Skin | kmader/skin-lesion-analysis | 3 GB | 10,015 |
| Blood | andrewmvd/leukemia-classification | 1 GB | 15,000 |
| ECG | shayanfazeli/heartbeat | 170 MB | 109,446 rows |
| Router | Built from above | — | ~1,600 |
