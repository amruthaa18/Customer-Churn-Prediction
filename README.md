# 📡 Customer Churn Prediction with Explainability

> An end-to-end Machine Learning project predicting telecom customer churn using XGBoost, SHAP explainability, and a Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

Customer churn is when a customer stops using a service. In telecom, retaining customers costs 5–7× less than acquiring new ones. This project builds a production-ready ML pipeline that:

- **Predicts** whether a customer will churn (binary classification)
- **Explains** every prediction using SHAP values
- **Provides** business-ready insights and retention recommendations
- **Deploys** as an interactive Streamlit web application

---

## 🎯 Business Objectives

| Objective | Description |
|-----------|-------------|
| Identify at-risk customers | Predict churn before it happens |
| Understand churn drivers | Find which factors cause churn |
| Enable proactive retention | Give business teams actionable insights |
| Build interpretable models | SHAP explanations for trust and compliance |

---

## 📊 Dataset

**Source**: [Telco Customer Churn — IBM Sample Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Size**: 7,043 customers × 21 features

| Feature | Type | Description |
|---------|------|-------------|
| customerID | ID | Unique identifier (dropped) |
| gender | Categorical | Male or Female |
| SeniorCitizen | Binary | Whether customer is 65+ |
| Partner | Binary | Has a partner |
| Dependents | Binary | Has dependents |
| tenure | Numerical | Months as customer |
| PhoneService | Binary | Has phone service |
| MultipleLines | Categorical | Single / multiple / no lines |
| InternetService | Categorical | DSL / Fiber optic / No |
| OnlineSecurity | Categorical | Has online security |
| OnlineBackup | Categorical | Has online backup |
| DeviceProtection | Categorical | Has device protection |
| TechSupport | Categorical | Has tech support |
| StreamingTV | Categorical | Streams TV |
| StreamingMovies | Categorical | Streams movies |
| Contract | Categorical | Month-to-month / One year / Two year |
| PaperlessBilling | Binary | Uses paperless billing |
| PaymentMethod | Categorical | 4 payment methods |
| MonthlyCharges | Numerical | Monthly bill in USD |
| TotalCharges | Numerical | Total billed amount |
| Churn | Target | Yes / No |

---

## 🏗️ Project Architecture

```
Raw CSV Data
    │
    ▼
┌─────────────────────────────┐
│   Data Cleaning             │  → Drop customerID, fix TotalCharges
│   Missing Value Handling    │  → ~11 rows with blank TotalCharges
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Feature Encoding          │  → Binary: Yes/No → 1/0
│   Label Encoding            │  → Multi-class categoricals
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   StandardScaler            │  → Normalize tenure, MonthlyCharges, TotalCharges
│   Train/Test Split (80/20)  │  → Stratified split
│   SMOTE Oversampling        │  → Balance 26/74 class imbalance
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Model Training            │  → LR, Random Forest, XGBoost
│   Hyperparameter Config     │  → class_weight, n_estimators, etc.
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Model Evaluation          │  → Accuracy, Precision, Recall, F1, ROC-AUC
│   Confusion Matrix          │  → Visualize TP/FP/TN/FN
│   ROC Curves                │  → Compare all models
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   SHAP Explainability       │  → Global importance (summary plot)
│   TreeExplainer             │  → Local explanation (waterfall)
│   Business Insights         │  → Actionable findings
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│   Streamlit Web App         │  → User input form
│                             │  → Probability display
│                             │  → SHAP charts
│                             │  → Business recommendations
└─────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save it as:
```
data/telco_churn.csv
```

### 4. Run EDA
```bash
python notebooks/01_EDA.py
```

### 5. Train Models + Generate SHAP
```bash
python notebooks/03_Modelling.py
```

### 6. Launch the App
```bash
streamlit run app/streamlit_app.py
```

---

## 📈 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.80 | 0.65 | 0.76 | 0.70 | 0.84 |
| Random Forest | 0.82 | 0.68 | 0.77 | 0.72 | 0.86 |
| **XGBoost** | **0.83** | **0.70** | **0.78** | **0.74** | **0.87** |

> XGBoost selected as the best model based on ROC-AUC score.

---

## 🔍 Key SHAP Findings

| Rank | Feature | Business Insight |
|------|---------|-----------------|
| 1 | Contract | Month-to-month → highest churn risk |
| 2 | tenure | Short tenure → new customers at risk |
| 3 | MonthlyCharges | High bills correlate with churn |
| 4 | TechSupport | No support → doubles churn probability |
| 5 | OnlineSecurity | No security → major churn driver |
| 6 | InternetService | Fiber optic users churn more |
| 7 | TotalCharges | Low total = new customer = high risk |

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── telco_churn.csv               # Raw dataset (download from Kaggle)
│
├── notebooks/
│   ├── 01_EDA.py                     # Full exploratory data analysis
│   ├── 03_Modelling.py               # Training + evaluation + SHAP pipeline
│   └── *.png                         # Generated plots
│
├── src/
│   ├── preprocess.py                 # Data cleaning, encoding, SMOTE
│   ├── train.py                      # Model training and saving
│   ├── evaluate.py                   # Metrics, confusion matrix, ROC
│   └── explain.py                    # SHAP explainability module
│
├── models/
│   ├── xgboost_model.pkl             # Trained XGBoost model
│   ├── random_forest_model.pkl       # Trained Random Forest
│   ├── logistic_regression_model.pkl # Trained Logistic Regression
│   └── scaler.pkl                    # Fitted StandardScaler
│
├── app/
│   └── streamlit_app.py              # Interactive web application
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── .gitignore                        # Git ignore rules
```

---

## 🧠 Techniques Used

- **Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Encoding**: Label Encoding for categorical features
- **Scaling**: StandardScaler for numerical features
- **Models**: Logistic Regression, Random Forest, XGBoost
- **Explainability**: SHAP TreeExplainer (global + local)
- **Evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Deployment**: Streamlit interactive web application

---

## 📝 License

MIT License — free for personal and commercial use.

---

## 🙏 Acknowledgements

- IBM Telco Customer Churn dataset
- SHAP library by Scott Lundberg
- XGBoost by Tianqi Chen
