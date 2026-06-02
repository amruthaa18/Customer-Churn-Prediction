# 📄 Resume & Interview Preparation
## Customer Churn Prediction with Explainability

---

## ✅ RESUME BULLET POINTS

Copy these directly onto your resume under Projects:

---

### Option A — Detailed Version

**Customer Churn Prediction with Explainability | Python, XGBoost, SHAP, Streamlit**
- Built an end-to-end binary classification pipeline to predict telecom customer churn using the IBM Telco dataset (7,043 records, 20 features), achieving ROC-AUC of 0.87 with XGBoost
- Engineered and compared 3 models (Logistic Regression, Random Forest, XGBoost); handled class imbalance using SMOTE, improving minority-class recall by ~18% over baseline
- Implemented SHAP TreeExplainer for global and local explainability; identified Contract type, tenure, and MonthlyCharges as top churn drivers with quantified business impact
- Deployed an interactive Streamlit web application allowing real-time churn probability scoring with per-prediction SHAP waterfall charts and automated retention recommendations

---

### Option B — Concise Version (1–2 lines)

**Customer Churn Prediction** | XGBoost · SHAP · SMOTE · Streamlit
- Predicted telecom customer churn (ROC-AUC 0.87) with full SHAP explainability pipeline and Streamlit deployment; revealed contract type and tenure as primary churn drivers

---

### Option C — Skills-Focused Version

**Customer Churn Prediction with XAI** | Python, scikit-learn, XGBoost, SHAP, Streamlit
- Applied SMOTE oversampling and StandardScaler preprocessing on imbalanced dataset (74/26 split); trained and evaluated 3 classifiers using Precision, Recall, F1, and ROC-AUC
- Used SHAP values to translate ML predictions into actionable business insights for retention teams; built end-to-end ML pipeline from raw CSV to deployed web application

---

## 🎤 INTERVIEW QUESTIONS & ANSWERS

---

### TOPIC 1: Problem Understanding

**Q1: What is customer churn and why is it important?**

A: Customer churn means a customer stops using a company's service. It is critical because retaining an existing customer is 5–7 times cheaper than acquiring a new one. By predicting churn in advance, companies can intervene with targeted offers or support before the customer leaves, protecting revenue.

---

**Q2: Why did you choose Recall as the primary metric instead of Accuracy?**

A: Because the dataset is imbalanced — about 74% of customers do not churn. A model that always predicts "No Churn" would get 74% accuracy but would fail to identify any actual churners, which defeats the business purpose. Recall measures what percentage of actual churners we correctly identified. In churn prediction, a false negative (missing a churner) is more costly than a false positive (offering a discount to someone who wasn't going to leave). So we prioritize Recall while keeping Precision reasonable, leading to an F1 score as the balanced metric.

---

### TOPIC 2: Data & Preprocessing

**Q3: How did you handle missing values in the dataset?**

A: The TotalCharges column had blank spaces instead of NaN for approximately 11 customers who had zero tenure. I replaced those spaces with NaN using pandas replace(), then converted the column to numeric using pd.to_numeric() with errors='coerce'. Since only 11 rows were affected out of 7,043 (0.15%), I dropped those rows — dropping such a small fraction has negligible impact on model performance.

---

**Q4: What is SMOTE and why did you use it?**

A: SMOTE stands for Synthetic Minority Over-sampling Technique. The dataset has a class imbalance — only 26% of customers churned. If we train directly on this imbalanced data, the model learns to be biased toward the majority class.

SMOTE creates synthetic (artificially generated) examples of the minority class by interpolating between existing minority class samples in feature space. Unlike simple duplication, SMOTE adds new information rather than repeating the same data. I applied SMOTE only to the training set, never the test set, to avoid data leakage. This improved Recall on the churn class by about 18%.

---

**Q5: Why did you use Label Encoding instead of One-Hot Encoding for categorical features?**

A: For tree-based models like XGBoost and Random Forest, Label Encoding works well because these models use split thresholds — the numeric value is just an index, and the tree learns appropriate splits. One-Hot Encoding would increase dimensionality significantly and is more beneficial for linear models that assume numerical relationships between feature values. For Logistic Regression in this project, we also used Label Encoding for simplicity and consistency across models.

---

### TOPIC 3: Modelling

**Q6: Why did you choose XGBoost as the final model?**

A: XGBoost outperformed Logistic Regression and Random Forest on ROC-AUC (0.87 vs 0.84 and 0.86). Beyond performance, XGBoost is:
- Fast to train due to parallel tree construction
- Handles missing values internally
- Built-in regularization (L1/L2) reduces overfitting
- Directly compatible with SHAP's TreeExplainer for fast, exact Shapley value computation
- Widely used in industry for tabular data — it has won hundreds of Kaggle competitions

---

**Q7: What is ROC-AUC and what does 0.87 mean?**

A: ROC stands for Receiver Operating Characteristic. The AUC is the Area Under the ROC Curve. The ROC curve plots True Positive Rate (Recall) against False Positive Rate at different classification thresholds.

An AUC of 0.87 means: if we randomly pick one churner and one non-churner from the dataset, there is an 87% probability that the model scores the churner higher than the non-churner. A score of 0.5 is random guessing; 1.0 is perfect. 0.87 is considered a strong result for this dataset.

---

**Q8: What is the confusion matrix and how do you interpret it?**

A: A confusion matrix is a 2×2 table showing:
- True Positives (TP): Predicted Churn = Churn, Actual = Churn ✅
- True Negatives (TN): Predicted No Churn, Actual = No Churn ✅
- False Positives (FP): Predicted Churn, Actual = No Churn ❌ (type I error)
- False Negatives (FN): Predicted No Churn, Actual = Churn ❌ (type II error)

For churn prediction, FN is more costly — we miss a customer who is about to leave. We minimize FN by optimizing Recall. FP leads to unnecessary retention offers, which is a small financial cost but much less damaging than losing the customer entirely.

---

### TOPIC 4: SHAP Explainability

**Q9: What is SHAP and why is it better than feature importance from the model itself?**

A: SHAP stands for SHapley Additive exPlanations. It comes from game theory — the idea is to fairly distribute the "credit" for a prediction among all features.

Built-in feature importance (like XGBoost's gain-based importance) only tells you which features the model used most across all trees — it is a global average and can be biased. SHAP values are:
- Consistent: if a feature has more impact, its SHAP value is always higher
- Local: each prediction has its own SHAP values explaining that specific case
- Directional: SHAP shows whether a feature pushed the prediction up or down
- Additive: all SHAP values sum to the actual prediction output

This makes SHAP much more trustworthy for business stakeholders.

---

**Q10: What is the difference between global and local SHAP explanations?**

A: Global SHAP explanation answers: "Which features matter most across all customers overall?" It aggregates the mean absolute SHAP values across the entire test set. The summary beeswarm plot also shows the direction — e.g., high monthly charges consistently push churn probability up across all customers.

Local SHAP explanation answers: "Why did the model predict THIS specific customer will churn?" The waterfall plot shows each feature's contribution for that individual — e.g., for customer #42, the prediction was driven primarily by month-to-month contract (+0.32) and no tech support (+0.18), offset slightly by long tenure (-0.09).

---

### TOPIC 5: Deployment & Business

**Q11: How does your Streamlit app work?**

A: The Streamlit app loads the pre-trained XGBoost model and fitted StandardScaler from disk. The user fills in a customer profile in the sidebar — demographics, services, contract type, and billing details. When the Predict button is clicked, the app encodes the inputs using the same mapping as training, applies the scaler to numerical features, runs the model, and displays:
1. Churn probability as a percentage with a color-coded risk indicator
2. A probability bar chart
3. A SHAP bar chart showing which features drove the prediction
4. Automated business recommendations based on the top risk factors

---

**Q12: If this model is deployed in production, what would you monitor?**

A: I would monitor:
1. Data drift — check if input feature distributions shift over time (e.g., new contract types are introduced)
2. Model drift — monitor if prediction accuracy degrades as customer behavior changes
3. Prediction distribution — track what percentage of customers are being flagged as high churn risk each month
4. Business outcome tracking — did customers flagged as churners actually churn? Calculate actual precision/recall monthly
5. Feature importance stability — ensure SHAP values remain consistent, indicating the model is still learning the same patterns

Tools I would use: MLflow for experiment tracking, Evidently AI or Arize for monitoring, and scheduled retraining pipelines.

---

## 🏷️ Technical Skills This Project Demonstrates

| Skill Category | What You Used |
|---------------|---------------|
| Data Analysis | pandas, numpy, EDA, univariate/bivariate analysis |
| Visualization | matplotlib, seaborn (histograms, boxplots, heatmaps, violin plots) |
| ML Preprocessing | Label Encoding, StandardScaler, train_test_split, SMOTE |
| ML Modelling | Logistic Regression, Random Forest, XGBoost |
| ML Evaluation | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| Explainability | SHAP TreeExplainer, global/local explanations |
| Deployment | Streamlit, joblib model serialization |
| Software Eng. | Modular code (src/), requirements.txt, .gitignore, README |
| Business Acumen | Translating ML output into retention strategy |
