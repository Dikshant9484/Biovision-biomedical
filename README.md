# 🔬 BioVision AI
**Biomedical Image Detection and Feature Extraction using ML/DL**

Full-stack AI healthcare app — React (Vite) frontend + Flask backend — for multi-modal cancer screening and ECG arrhythmia detection.

---

## 📁 Project Structure

```
biovision-ai/
│
├── backend/                        Flask REST API
│   ├── app.py                      Entry point
│   ├── requirements.txt
│   ├── routes/
│   │   ├── breast.py               POST /api/predict/breast-image
│   │   │                           POST /api/predict/breast-tabular
│   │   ├── skin.py                 POST /api/predict/skin
│   │   ├── blood.py                POST /api/predict/blood
│   │   ├── ecg.py                  POST /api/predict/ecg
│   │   ├── universal.py            POST /api/predict/universal
│   │   ├── chat.py                 POST /api/chat/recommendation
│   │   └── health.py               GET  /api/health
│   ├── services/
│   │   ├── breast_service.py       ResNet50 + tabular MLP inference
│   │   ├── skin_service.py         ResNet50 + ABCDE features
│   │   ├── blood_service.py        ResNet50 leukemia detection
│   │   ├── ecg_service.py          GCN pipeline (CSV → graph → predict)
│   │   ├── universal_service.py    Router + specialist pipeline
│   │   └── chat_service.py         Groq LLaMA3 chatbot
│   ├── models/
│   │   ├── model_loader.py         Centralized lazy model loading
│   │   └── weights/                ← place .h5 / .pt / .pkl here
│   └── utils/
│       ├── image_utils.py          Preprocessing helpers
│       └── feature_extractor.py    Image + ECG feature extraction
│
├── frontend/                       React + Vite + Tailwind (single page)
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx                 ← entire UI in one file
│       └── index.css
│
├── training/                       Model training scripts
│   ├── breast_train.py             Tabular MLP (sklearn dataset, no download)
│   ├── breast_image_train.py       ResNet50 mammogram
│   ├── skin_train.py               ResNet50 HAM10000
│   ├── blood_train.py              ResNet50 ALL leukemia
│   ├── ecg_train.py                GCN MIT-BIH arrhythmia
│   └── router_train.py             Universal router classifier
│
├── render.yaml                     Render deployment config
├── .env.example                    Environment variable template
├── .gitignore
└── README.md
```

---

## 🚀 Local Setup

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp ../.env.example .env
# Open .env and add your GROQ_API_KEY from https://console.groq.com

# (Optional) Copy existing model weights
# cp path/to/breast_model.h5   models/weights/breast_model.h5
# cp path/to/breast_scaler.pkl models/weights/breast_scaler.pkl

# Start server
python app.py
# → http://localhost:5000
# → Test: http://localhost:5000/api/health
```

> **Note:** The app runs in **mock/demo mode** if model weights are not found.
> All endpoints still return results using heuristics so you can test the UI immediately.

### 2. Frontend

```bash
cd frontend

# Install
npm install

# Set API URL (points to Flask in dev)
cp .env.example .env
# VITE_API_URL is already set to /api — Vite proxies to localhost:5000

# Start
npm run dev
# → http://localhost:5173
```

---

## 🏋️ Training Models

### Breast Tabular (no dataset download needed)
```bash
cd backend
python ../training/breast_train.py
# Saves: models/weights/breast_model.h5 + breast_scaler.pkl
```

### Skin Cancer — HAM10000
```bash
# 1. Download: https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection
# 2. Place images:
#    datasets/skin/train/benign/  datasets/skin/train/malignant/
#    datasets/skin/val/benign/    datasets/skin/val/malignant/
python training/skin_train.py
```

### Blood Cancer — ALL Leukemia
```bash
# 1. Download: https://www.kaggle.com/datasets/andrewmvd/leukemia-classification
# 2. Place images:
#    datasets/blood/train/normal/    datasets/blood/train/leukemia/
#    datasets/blood/val/normal/      datasets/blood/val/leukemia/
python training/blood_train.py
```

### ECG GCN — MIT-BIH
```bash
# 1. Download: https://www.kaggle.com/datasets/shayanfazeli/heartbeat
# 2. Place: datasets/ecg/mitbih_train.csv  datasets/ecg/mitbih_test.csv
python training/ecg_train.py
```

---

## 🌐 Deploy to Render

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "BioVision AI"
git remote add origin https://github.com/yourusername/biovision-ai
git push -u origin main
```

### Step 2 — Create services on Render
1. Go to [render.com](https://render.com) → **New → Blueprint**
2. Connect your GitHub repo
3. Render reads `render.yaml` and creates **both services automatically**

### Step 3 — Set environment variables

**Backend service** → Environment tab:
```
GROQ_API_KEY   = your_groq_api_key_here
```

**Frontend service** → Environment tab:
```
VITE_API_URL   = https://biovision-backend.onrender.com/api
```

Also update `ALLOWED_ORIGINS` in the backend to match your frontend URL.

### Step 4 — Model weights on Render

Render's free tier doesn't persist large files. Options:
- **Option A:** Commit weights to Git if each file < 100 MB
- **Option B:** Use Render Disks (paid) and set `*_MODEL_PATH` env vars
- **Option C:** Download from Hugging Face Hub at startup (add to `app.py`)

---

## 🔌 API Reference

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| GET  | `/api/health` | — | Health check |
| POST | `/api/predict/breast-image` | `file` (image) | Breast image classification |
| POST | `/api/predict/breast-tabular` | `{ features: [30 floats] }` | Clinical risk estimator |
| POST | `/api/predict/skin` | `file` (image) | Skin cancer |
| POST | `/api/predict/blood` | `file` (image) | Blood cancer |
| POST | `/api/predict/ecg` | `file` (.csv) | ECG GCN arrhythmia |
| POST | `/api/predict/universal` | `file` (image) | Auto-detect + classify |
| POST | `/api/chat/recommendation` | `{ prediction_context, user_message, chat_history }` | Groq chatbot |

---

## ⚠️ Disclaimer

BioVision AI is an **academic project** for educational purposes only.
It is a screening support tool and is **NOT a substitute for professional medical diagnosis.**
Always consult qualified healthcare providers.
