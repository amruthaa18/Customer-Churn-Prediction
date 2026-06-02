"""
streamlit_app.py
----------------
Interactive web app for Customer Churn Prediction.
Allows user input, shows churn probability, and explains prediction with SHAP.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #2196F3;
    }
    .churn-risk-high {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .churn-risk-low {
        background: linear-gradient(135deg, #55efc4, #00b894);
        color: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .insight-box {
        background: #e3f2fd;
        border-left: 4px solid #1976D2;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Scaler ──────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the trained model and scaler. Cached for performance."""
    model_path = 'models/xgboost_model.pkl'
    scaler_path = 'models/scaler.pkl'

    if not os.path.exists(model_path):
        st.error("⚠️ Model not found. Please run the training pipeline first.")
        st.code("python notebooks/03_Modelling.py", language="bash")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_artifacts()


# ── Feature Names ─────────────────────────────────────────────
FEATURE_NAMES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

# Encoding maps (must match preprocessing)
ENCODE_MAP = {
    'gender': {'Male': 1, 'Female': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
    'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
    'OnlineSecurity': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'OnlineBackup': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'DeviceProtection': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'TechSupport': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingTV': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'StreamingMovies': {'No': 0, 'No internet service': 1, 'Yes': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaymentMethod': {
        'Bank transfer (automatic)': 0,
        'Credit card (automatic)': 1,
        'Electronic check': 2,
        'Mailed check': 3
    },
}


def preprocess_input(user_input: dict) -> np.ndarray:
    """Convert user input dict to model-ready array."""
    row = {}
    for feat in FEATURE_NAMES:
        val = user_input[feat]
        if feat in ENCODE_MAP:
            row[feat] = ENCODE_MAP[feat][val]
        else:
            row[feat] = val  # numerical

    df_row = pd.DataFrame([row])

    # Scale numerical columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_row[num_cols] = scaler.transform(df_row[num_cols])

    return df_row.values


# ── Sidebar: Customer Input Form ─────────────────────────────
st.sidebar.title("📋 Customer Profile")
st.sidebar.markdown("Fill in the customer details below:")

with st.sidebar:
    st.subheader("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    st.subheader("📱 Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multi_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
    backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])
    device = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
    tech = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
    movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])

    st.subheader("💳 Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Bank transfer (automatic)", "Credit card (automatic)",
        "Electronic check", "Mailed check"
    ])

    st.subheader("💰 Charges")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
    total = st.number_input("Total Charges ($)", min_value=0.0, value=float(monthly * tenure), step=10.0)

    predict_btn = st.button("🔍 Predict Churn", use_container_width=True, type="primary")


# ── Main Panel ───────────────────────────────────────────────
st.title("📡 Customer Churn Prediction Dashboard")
st.markdown("Powered by XGBoost + SHAP Explainability")

if predict_btn:
    user_input = {
        'gender': gender, 'SeniorCitizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone,
        'MultipleLines': multi_lines, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup,
        'DeviceProtection': device, 'TechSupport': tech,
        'StreamingTV': tv, 'StreamingMovies': movies,
        'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment, 'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    X_input = preprocess_input(user_input)
    churn_prob = model.predict_proba(X_input)[0][1]
    churn_pred = int(churn_prob >= 0.5)

    # ── Prediction Result ────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if churn_pred == 1:
            st.markdown(f"""
            <div class="churn-risk-high">
                🚨 HIGH CHURN RISK<br>
                <span style="font-size:36px">{churn_prob*100:.1f}%</span><br>
                <span style="font-size:14px">Probability of Churn</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="churn-risk-low">
                ✅ LOW CHURN RISK<br>
                <span style="font-size:36px">{churn_prob*100:.1f}%</span><br>
                <span style="font-size:14px">Probability of Churn</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Contract Type", contract)
        st.metric("Tenure", f"{tenure} months")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Monthly Charges", f"${monthly:.2f}")
        st.metric("Internet Service", internet)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Probability Gauge ────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Churn Probability Breakdown")

    fig, ax = plt.subplots(figsize=(8, 1.5))
    ax.barh(["No Churn", "Churn"],
            [1 - churn_prob, churn_prob],
            color=['#4CAF50', '#F44336'], edgecolor='white', height=0.5)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Decision boundary (0.5)')
    ax.set_xlabel("Probability")
    ax.set_title("Prediction Probability", fontweight='bold')
    for i, v in enumerate([1 - churn_prob, churn_prob]):
        ax.text(v + 0.01, i, f"{v*100:.1f}%", va='center', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── SHAP Explanation ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🔬 Why This Prediction? (SHAP Explanation)")

    with st.spinner("Computing SHAP values..."):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_input)

        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap_val = vals[0]

        # Top features affecting this prediction
        shap_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'SHAP Value': shap_val,
            'Impact': ['↑ Churn' if v > 0 else '↓ Churn' for v in shap_val]
        }).reindex(pd.Series(np.abs(shap_val)).sort_values(ascending=False).index)

        top_n = 10
        top_shap = shap_df.head(top_n)

        col_a, col_b = st.columns([1.2, 1])

        with col_a:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            colors = ['#F44336' if v > 0 else '#4CAF50' for v in top_shap['SHAP Value']]
            bars = ax2.barh(top_shap['Feature'], top_shap['SHAP Value'],
                            color=colors, edgecolor='white')
            ax2.axvline(0, color='black', linewidth=0.8)
            ax2.set_xlabel("SHAP Value (impact on churn probability)")
            ax2.set_title("Top Feature Contributions", fontweight='bold')
            ax2.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        with col_b:
            st.markdown("**Feature Impact Summary**")
            for _, row in top_shap.iterrows():
                icon = "🔴" if row['SHAP Value'] > 0 else "🟢"
                st.markdown(f"{icon} **{row['Feature']}**: {row['SHAP Value']:+.4f} ({row['Impact']})")

    # ── Business Recommendation ──────────────────────────────
    st.markdown("---")
    st.subheader("💡 Business Recommendation")

    recs = []
    if contract == "Month-to-month":
        recs.append("📋 **Offer a discounted annual or two-year contract** — month-to-month customers churn 10x more.")
    if tech == "No":
        recs.append("🛠️ **Bundle tech support** — customers without it are twice as likely to churn.")
    if security == "No":
        recs.append("🔒 **Offer online security add-on** — a major churn driver.")
    if internet == "Fiber optic" and monthly > 80:
        recs.append("💰 **Consider a loyalty discount** — high-charge fiber users are at high risk.")
    if tenure < 12:
        recs.append("🎁 **New customer care package** — most churn happens in first year.")
    if payment == "Electronic check":
        recs.append("💳 **Encourage auto-pay setup** — electronic check users churn more.")

    if not recs:
        recs.append("✅ This customer appears stable. Continue standard engagement.")

    for r in recs:
        st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)

else:
    # Default landing content
    st.markdown("""
    ## How to use this app

    1. **Fill in the customer profile** in the left sidebar
    2. **Click "Predict Churn"** to get the prediction
    3. **Review the SHAP explanation** to understand why
    4. **Act on the business recommendations** shown at the bottom

    ---
    ### About this model
    - **Algorithm**: XGBoost Classifier
    - **Training Data**: Telco Customer Churn dataset (7,043 customers)
    - **Explainability**: SHAP (SHapley Additive exPlanations)
    - **Key Metrics**: ROC-AUC ~0.87, F1 ~0.82

    ### Key Churn Drivers Found
    | Factor | Impact |
    |--------|--------|
    | Month-to-month contract | 🔴 Very High |
    | Short tenure (< 12 months) | 🔴 High |
    | No tech support | 🔴 High |
    | High monthly charges | 🔴 High |
    | No online security | 🟠 Medium |
    | Electronic check payment | 🟠 Medium |
    | Senior citizen | 🟡 Moderate |
    | Two-year contract | 🟢 Protective |
    """)
