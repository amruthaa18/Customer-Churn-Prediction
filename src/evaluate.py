"""
evaluate.py
-----------
Evaluates trained models using multiple metrics.
Generates confusion matrix, ROC curve, and comparison report.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
import os


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Evaluate a single trained model.
    Returns a dictionary of all metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of churn

    metrics = {
        "Model": model_name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall":    round(recall_score(y_test, y_pred), 4),
        "F1 Score":  round(f1_score(y_test, y_pred), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics


def evaluate_all_models(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and return a comparison DataFrame.
    """
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(metrics)

    df_results = pd.DataFrame(results).set_index("Model")
    print("\n[COMPARISON TABLE]")
    print(df_results.to_string())
    return df_results


def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Model", save_path: str = None):
    """Plot and optionally save a styled confusion matrix."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
        linewidths=0.5, linecolor='gray'
    )
    plt.title(f'Confusion Matrix — {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[SAVED] Confusion matrix → {save_path}")
    plt.show()


def plot_roc_curves(trained_models: dict, X_test, y_test, save_path: str = None):
    """Plot ROC curves for all models on one chart."""
    plt.figure(figsize=(8, 6))

    colors = ['#2196F3', '#4CAF50', '#F44336']
    for (name, model), color in zip(trained_models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', color=color, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[SAVED] ROC curves → {save_path}")
    plt.show()


def plot_metrics_comparison(df_results: pd.DataFrame, save_path: str = None):
    """Bar chart comparing all models across all metrics."""
    df_plot = df_results[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']]

    ax = df_plot.plot(
        kind='bar', figsize=(10, 6),
        color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336'],
        edgecolor='white', width=0.7
    )
    plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=7, padding=2)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
