"""
train.py
--------
Trains Logistic Regression, Random Forest, and XGBoost models.
Saves the best model (XGBoost) to disk.
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from preprocess import run_preprocessing_pipeline


def get_models() -> dict:
    """
    Returns a dictionary of model name → model instance.
    All models are configured with sensible defaults.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'   # handles imbalance without SMOTE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1                 # use all CPU cores
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }
    return models


def train_all_models(X_train, y_train) -> dict:
    """
    Train all three models on the training data.
    Returns dictionary of model name → trained model.
    """
    models = get_models()
    trained = {}

    for name, model in models.items():
        print(f"[TRAINING] {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"[DONE]     {name} trained successfully.")

    return trained


def save_model(model, path: str):
    """Save a trained model to disk using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[SAVED] Model saved to {path}")


def load_model(path: str):
    """Load a saved model from disk."""
    model = joblib.load(path)
    print(f"[LOADED] Model loaded from {path}")
    return model


if __name__ == '__main__':
    # Run full pipeline
    X_train, X_test, y_train, y_test, features = run_preprocessing_pipeline(
        filepath='data/telco_churn.csv',
        scaler_path='models/scaler.pkl',
        use_smote=True
    )

    # Train all models
    trained_models = train_all_models(X_train, y_train)

    # Save the XGBoost model (best performer typically)
    save_model(trained_models["XGBoost"], 'models/xgboost_model.pkl')
    save_model(trained_models["Random Forest"], 'models/random_forest_model.pkl')
    save_model(trained_models["Logistic Regression"], 'models/logistic_regression_model.pkl')

    print("\n[COMPLETE] All models trained and saved.")
