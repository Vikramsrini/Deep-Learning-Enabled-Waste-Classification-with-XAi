Project structure prepared for the requested deliverables.

Contents are organized into three categories:

1. Hybrid model code without optimization including analysis
2. Hybrid model with optimization including analysis
3. Explainable AI only

Target hybrid models only:
- EfficientNetV2-S + LeViT-256 + XGBoost
- EfficientNetV2-S + Swin-Base + XGBoost

Included files:

1. Without optimization including analysis
- `1_without_optimization_including_analysis/efficientnetv2s_levit256_xgboost_without_optimization_with_analysis.ipynb`
- `1_without_optimization_including_analysis/efficientnetv2s_swinbase_xgboost_without_optimization_with_analysis.ipynb`

2. With optimization including analysis
- `2_with_optimization_including_analysis/efficientnetv2s_levit256_xgboost_best_available_with_analysis.ipynb`
- `2_with_optimization_including_analysis/efficientnetv2s_swinbase_xgboost_optuna_with_analysis.ipynb`
- `2_with_optimization_including_analysis/launch_hpo.py`
- `2_with_optimization_including_analysis/xgboost_optimizer.py`

3. Explainable AI only
- `3_explainable_ai_only/efficientnetv2s_levit256_xgboost_explainable_ai.ipynb`
- `3_explainable_ai_only/efficientnetv2s_swinbase_xgboost_explainable_ai.ipynb`
