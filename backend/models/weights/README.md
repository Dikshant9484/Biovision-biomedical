# BioVision AI — Model Weights Directory

Place your trained model weights here before running the backend.

## Required Files

| File | Description | Source |
|------|-------------|--------|
| `breast_model.h5` | Breast tabular MLP | Run `training/breast_train.py` |
| `breast_scaler.pkl` | StandardScaler for breast | Run `training/breast_train.py` |
| `breast_image_model.h5` | Breast ResNet50 | Run `training/breast_image_train.py` |
| `skin_model.h5` | Skin ResNet50 (HAM10000) | Run `training/skin_train.py` |
| `blood_model.h5` | Blood ResNet50 (ALL dataset) | Run `training/blood_train.py` |
| `lung_model.h5` | Lung ResNet50 | Run `training/lung_train.py` |
| `router_model.h5` | Universal router classifier | Run `training/router_train.py` |
| `ecg_gcn.pt` | ECG GCN PyTorch model | Run `training/ecg_train.py` |

## Notes
- The app runs in **mock/demo mode** if weights are not found.
- Mock mode uses heuristics and random values for demo purposes.
- For production accuracy, train each model with its dataset.

## Existing weights from your project
Copy these to this directory:
```
breast_model.h5      → models/weights/breast_model.h5
breast_scaler.pkl    → models/weights/breast_scaler.pkl
fully_trainable_lung_model.h5 → models/weights/lung_model.h5
```
