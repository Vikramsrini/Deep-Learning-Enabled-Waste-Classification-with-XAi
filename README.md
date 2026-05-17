# Title

deep learning enabled waste classification with xai

# Description

This repository contains code for two hybrid waste-classification models and their analysis tracks:

- EfficientNetV2-S + LeViT-256 + XGBoost
- EfficientNetV2-S + Swin-Base + XGBoost

The deliverables are organized into:

1. Without optimization including analysis
2. With optimization including analysis
3. Explainable AI only

# Dataset Information

The experiments use the **TrashNeXt dataset** [30], which is specifically designed for automatic waste-material classification.

- **Total images:** 23,625 RGB images
- **Classes (9):** Cardboard, E-Waste, Foam Rubber, Glass, Medical, Metal, Organic, Paper, Plastic
- The classes cover household, industrial, and institutional waste streams, including recyclable and non-recyclable materials.

The dataset is pre-divided into three non-overlapping, independently sampled splits:

- **Train:** 18,898 images (80%)
- **Validation:** 2,363 images (10%)
- **Test:** 2,364 images (10%)

This split design preserves class representation reasonably well and prevents data leakage between training, hyperparameter optimization, and final evaluation.

### TrashNeXt Dataset — Class-Level Image Distribution Across Splits

| Waste Class | Train | Validation | Test | Category Type |
|---|---:|---:|---:|---|
| Cardboard | 1,886 | 236 | 235 | Recyclable / Dry |
| E-Waste | 2,404 | 301 | 301 | Hazardous / Electronic |
| Foam Rubber | 2,289 | 287 | 287 | Non-Recyclable / Soft |
| Glass | 2,009 | 251 | 252 | Recyclable / Rigid |
| Medical | 1,565 | 196 | 196 | Hazardous / Regulated |
| Metal | 2,065 | 258 | 258 | Recyclable / Rigid |
| Organic | 2,391 | 299 | 299 | Compostable / Biodegradable |
| Paper | 2,155 | 269 | 270 | Recyclable / Dry |
| Plastic | 2,135 | 267 | 267 | Recyclable / Flexible/Rigid |
| **TOTAL** | **18,898** | **2,363** | **2,364** | **23,625 total images** |

The dataset shows **moderate class imbalance**: E-Waste and Organic have the highest sample counts, while Medical has the lowest. This supports using weighted and macro-aware evaluation metrics (e.g., macro F1-score and weighted AUROC) rather than relying only on plain accuracy.

# Code Information

## 1) Without optimization including analysis

- `1_without_optimization_including_analysis/efficientnetv2s_levit256_xgboost_without_optimization_with_analysis.ipynb`
- `1_without_optimization_including_analysis/efficientnetv2s_swinbase_xgboost_without_optimization_with_analysis.ipynb`

## 2) With optimization including analysis

- `2_with_optimization_including_analysis/efficientnetv2s_levit256_xgboost_best_available_with_analysis.ipynb`
- `2_with_optimization_including_analysis/efficientnetv2s_swinbase_xgboost_optuna_with_analysis.ipynb`
- `2_with_optimization_including_analysis/xgboost_optimizer.py`
- `2_with_optimization_including_analysis/launch_hpo.py`

## 3) Explainable AI only

- `3_explainable_ai_only/efficientnetv2s_levit256_xgboost_explainable_ai.ipynb`
- `3_explainable_ai_only/efficientnetv2s_swinbase_xgboost_explainable_ai.ipynb`

# Usage Instructions

1. Prepare the dataset in class-wise train/validation/test folders.
2. Open the needed notebook in Jupyter Notebook or Google Colab.
3. Update dataset paths if your environment differs.
4. Install required Python dependencies.
5. Run notebook cells in order to perform feature extraction, XGBoost training/tuning, evaluation, and explainability where applicable.

For optimization scripts:

```bash
python 2_with_optimization_including_analysis/xgboost_optimizer.py
python 2_with_optimization_including_analysis/launch_hpo.py
```

# Requirements

Main dependencies used across the notebooks/scripts:

- Python 3.x
- torch
- torchvision
- timm
- numpy
- pandas
- xgboost
- scikit-learn
- matplotlib
- shap
- optuna
- clearml
- Pillow

Example install:

```bash
pip install torch torchvision timm numpy pandas xgboost scikit-learn matplotlib shap optuna clearml pillow
```

# Methodology

1. Load and preprocess waste image data.
2. Extract features from EfficientNetV2-S and either LeViT-256 or Swin-Base.
3. Concatenate hybrid features.
4. Train XGBoost classifier.
5. Evaluate with classification metrics and analysis plots.
6. In optimization track, run hyperparameter optimization workflows.
7. In explainable AI track, use SHAP-based interpretation.

# Citations

If used in research, cite the project/paper and major frameworks:

- XGBoost
- Optuna
- SHAP
- PyTorch
- TIMM

# License & Contribution Guidelines

- License: MIT (see `LICENSE`).
- Contributions should keep the repository focused on the two requested hybrid models and the three deliverable tracks.
