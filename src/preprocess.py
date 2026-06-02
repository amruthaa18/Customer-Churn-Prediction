"""
preprocess.py
-------------
Handles all data preprocessing for the Telco Customer Churn project.
Steps: load data, clean, encode, scale, handle class imbalance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data:
    - Replace blank strings with NaN
    - Convert TotalCharges to numeric
    - Drop customerID (not a feature)
    - Drop rows with missing values
    """
    df = df.copy()

    # customerID is just an identifier, not useful for prediction
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # TotalCharges has spaces as missing values in the raw dataset
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Drop rows where TotalCharges is NaN (only ~11 rows)
    before = df.shape[0]
    df.dropna(inplace=True)
    after = df.shape[0]
    print(f"[INFO] Dropped {before - after} rows with missing values. Remaining: {after}")

    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features:
    - Binary columns (Yes/No) → 1/0
    - Multi-class columns → Label Encoding
    - Target column 'Churn' → 1/0
    """
    df = df.copy()

    # Binary Yes/No columns → 1/0
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'Churn'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    # gender → 1/0
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Multi-value categorical columns → Label Encoding
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaymentMethod'
    ]
    le = LabelEncoder()
    for col in multi_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    print("[INFO] Encoding complete.")
    return df


def scale_features(df: pd.DataFrame, fit: bool = True, scaler_path: str = None):
    """
    Scale continuous numerical features using StandardScaler.
    If fit=True, fit and save the scaler.
    If fit=False, load and transform using saved scaler.
    """
    df = df.copy()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    if fit:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            print(f"[INFO] Scaler saved to {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[num_cols] = scaler.transform(df[num_cols])
        print(f"[INFO] Scaler loaded from {scaler_path}")

    print("[INFO] Scaling complete.")
    return df


def split_data(df: pd.DataFrame, target: str = 'Churn', test_size: float = 0.2, random_state: int = 42):
    """Split into train and test sets (80/20)."""
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"[INFO] Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to balance the training set.
    SMOTE = Synthetic Minority Over-sampling Technique.
    It creates synthetic samples for the minority class (Churn=1)
    instead of just duplicating existing ones.
    """
    print(f"[INFO] Before SMOTE — Class distribution: {dict(y_train.value_counts())}")
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE  — Class distribution: {dict(pd.Series(y_resampled).value_counts())}")
    return X_resampled, y_resampled


def run_preprocessing_pipeline(filepath: str, scaler_path: str = 'models/scaler.pkl', use_smote: bool = True):
    """
    Full preprocessing pipeline from raw CSV to train/test arrays.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df, fit=True, scaler_path=scaler_path)
    X_train, X_test, y_train, y_test = split_data(df)

    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    feature_names = [c for c in df.columns if c != 'Churn']
    print(f"[INFO] Features used: {feature_names}")
    return X_train, X_test, y_train, y_test, feature_names


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, features = run_preprocessing_pipeline(
        filepath='data/telco_churn.csv',
        scaler_path='models/scaler.pkl'
    )
    print("[DONE] Preprocessing pipeline complete.")
