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

The repository does not contain raw dataset files. The notebooks expect an image dataset arranged using class-wise folders under train/validation/test directories.

Typical expected paths in notebooks:

- `/content/dataset/Train`
- `/content/dataset/Valid`
- `/content/dataset/Test`

Some scripts in the optimization track also expect precomputed NumPy arrays (`X_train.npy`, `y_train.npy`, `X_valid.npy`, `y_valid.npy`) when using ClearML-based workflows.

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
