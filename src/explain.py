"""
explain.py
----------
SHAP-based Explainable AI module.
Provides global feature importance and local (per-prediction) explanations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os


def get_shap_explainer(model, X_train, model_type: str = "xgboost"):
    """
    Create the right SHAP explainer for the model type.

    - TreeExplainer: fast, exact for tree models (XGBoost, Random Forest)
    - LinearExplainer: for Logistic Regression
    """
    if model_type in ["xgboost", "random_forest"]:
        explainer = shap.TreeExplainer(model)
    elif model_type == "logistic_regression":
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))

    print(f"[INFO] SHAP explainer created for {model_type}")
    return explainer


def compute_shap_values(explainer, X_test):
    """Compute SHAP values for the test set."""
    shap_values = explainer.shap_values(X_test)
    print(f"[INFO] SHAP values computed. Shape: {np.array(shap_values).shape}")
    return shap_values


def plot_global_importance(shap_values, X_test, feature_names, save_path: str = None):
    """
    Global Feature Importance — SHAP Summary Plot (Bar).
    Shows which features matter most ACROSS all predictions.

    Business insight: tells us what drives churn in general.
    """
    plt.figure(figsize=(10, 7))

    # For binary classification with TreeExplainer, shap_values may be a list [class0, class1]
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    shap.summary_plot(
        vals,
        pd.DataFrame(X_test, columns=feature_names),
        plot_type="bar",
        show=False,
        color='#2196F3'
    )
    plt.title("Global Feature Importance (SHAP)", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Global importance → {save_path}")
    plt.show()


def plot_shap_summary_beeswarm(shap_values, X_test, feature_names, save_path: str = None):
    """
    SHAP Beeswarm / Dot Plot.
    Shows direction AND magnitude of each feature's impact.
    Red = high feature value, Blue = low feature value.

    Business insight: e.g., high monthly charges pushes toward churn.
    """
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        vals,
        pd.DataFrame(X_test, columns=feature_names),
        show=False
    )
    plt.title("SHAP Beeswarm Plot — Feature Impact Direction", fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Beeswarm plot → {save_path}")
    plt.show()


def explain_single_prediction(explainer, shap_values, X_test, feature_names,
                               index: int = 0, save_path: str = None):
    """
    Local Explanation — SHAP Waterfall/Force Plot for ONE customer.
    Explains why the model predicted churn or no-churn for that specific customer.

    Business insight: useful for customer service agent explaining to a manager.
    """
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values

    sample = pd.DataFrame(X_test, columns=feature_names).iloc[index]
    shap_val = vals[index]

    print(f"\n[LOCAL EXPLANATION] Customer index: {index}")
    print("Top 5 features pushing this prediction:")
    feature_impact = pd.Series(shap_val, index=feature_names).abs().sort_values(ascending=False)
    print(feature_impact.head(5).to_string())

    # Force plot (saved as HTML or matplotlib)
    plt.figure()
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_val,
            base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list)
                        else explainer.expected_value,
            data=sample.values,
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[SAVED] Local explanation → {save_path}")
    plt.show()


def get_top_shap_features(shap_values, feature_names, top_n: int = 10) -> pd.DataFrame:
    """
    Returns a DataFrame of top N features by mean absolute SHAP value.
    Useful for reporting and the Streamlit app.
    """
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    mean_abs = np.abs(vals).mean(axis=0)
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs
    }).sort_values('Mean |SHAP|', ascending=False).head(top_n)
    return df


def print_business_insights(shap_df: pd.DataFrame):
    """
    Translate SHAP results into plain business language.
    """
    print("\n" + "="*60)
    print("  BUSINESS INSIGHTS FROM SHAP ANALYSIS")
    print("="*60)
    insights = {
        'tenure':          "Customers with shorter tenure are at higher churn risk — focus retention efforts on new customers.",
        'MonthlyCharges':  "High monthly charges significantly increase churn probability — consider loyalty discounts.",
        'Contract':        "Month-to-month contracts are major churn drivers — incentivize annual/two-year contracts.",
        'TechSupport':     "Lack of tech support is associated with churn — promote tech support bundles.",
        'OnlineSecurity':  "Customers without online security churn more — include security in starter plans.",
        'InternetService': "Fiber optic customers show higher churn — investigate service quality and pricing.",
        'TotalCharges':    "Total charges reflect tenure × monthly spend — longer-term high spenders need VIP treatment.",
    }
    for feature, insight in insights.items():
        if feature in shap_df['Feature'].values:
            rank = shap_df[shap_df['Feature'] == feature].index[0] + 1
            print(f"\n  #{rank} {feature.upper()}")
            print(f"     → {insight}")
    print("\n" + "="*60)
