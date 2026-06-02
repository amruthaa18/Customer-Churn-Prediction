"""
03_Modelling.py
---------------
Full modelling pipeline: preprocessing → training → evaluation → SHAP.
Run this from the project root: python notebooks/03_Modelling.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from preprocess import run_preprocessing_pipeline
from train import train_all_models, save_model
from evaluate import evaluate_all_models, plot_confusion_matrix, plot_roc_curves, plot_metrics_comparison
from explain import (get_shap_explainer, compute_shap_values,
                     plot_global_importance, plot_shap_summary_beeswarm,
                     explain_single_prediction, get_top_shap_features, print_business_insights)

print("="*60)
print("  CUSTOMER CHURN PREDICTION — FULL PIPELINE")
print("="*60)

# ── STEP 1: Preprocessing ────────────────────────────────────
print("\n[STEP 1] Running preprocessing pipeline...")
X_train, X_test, y_train, y_test, feature_names = run_preprocessing_pipeline(
    filepath='data/telco_churn.csv',
    scaler_path='models/scaler.pkl',
    use_smote=True
)

# ── STEP 2: Train Models ─────────────────────────────────────
print("\n[STEP 2] Training all models...")
trained_models = train_all_models(X_train, y_train)

# Save models
os.makedirs('models', exist_ok=True)
for name, model in trained_models.items():
    fname = name.lower().replace(' ', '_') + '_model.pkl'
    save_model(model, f'models/{fname}')

# ── STEP 3: Evaluate ─────────────────────────────────────────
print("\n[STEP 3] Evaluating all models...")
df_results = evaluate_all_models(trained_models, X_test, y_test)

# Plots
plot_roc_curves(trained_models, X_test, y_test, save_path='notebooks/roc_curves.png')
plot_metrics_comparison(df_results, save_path='notebooks/metrics_comparison.png')

best_model_name = df_results['ROC-AUC'].idxmax()
best_model = trained_models[best_model_name]
print(f"\n[BEST MODEL] {best_model_name} with ROC-AUC = {df_results.loc[best_model_name, 'ROC-AUC']}")

plot_confusion_matrix(best_model, X_test, y_test,
                      model_name=best_model_name,
                      save_path=f'notebooks/confusion_matrix_{best_model_name.replace(" ","_")}.png')

# ── STEP 4: SHAP Explainability ───────────────────────────────
print("\n[STEP 4] Computing SHAP values...")

# Use XGBoost for SHAP (most interpretable with TreeExplainer)
xgb_model = trained_models["XGBoost"]
explainer = get_shap_explainer(xgb_model, X_train, model_type="xgboost")
shap_values = compute_shap_values(explainer, X_test)

plot_global_importance(shap_values, X_test, feature_names,
                       save_path='notebooks/shap_global_importance.png')

plot_shap_summary_beeswarm(shap_values, X_test, feature_names,
                            save_path='notebooks/shap_beeswarm.png')

# Local explanation for a churning customer
y_test_arr = np.array(y_test)
churn_indices = np.where(y_test_arr == 1)[0]
explain_single_prediction(explainer, shap_values, X_test, feature_names,
                           index=churn_indices[0],
                           save_path='notebooks/shap_local_explanation.png')

# Top features + business insights
shap_df = get_top_shap_features(shap_values, feature_names, top_n=10)
print("\n[TOP 10 SHAP FEATURES]")
print(shap_df.to_string(index=False))

print_business_insights(shap_df)

print("\n" + "="*60)
print("  PIPELINE COMPLETE. All outputs saved to notebooks/")
print("="*60)
