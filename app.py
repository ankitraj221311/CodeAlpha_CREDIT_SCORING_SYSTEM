import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Credit Scoring System",
    page_icon="💳",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoders = joblib.load("models/encoders.pkl")

# =========================
# TITLE
# =========================

st.title("💳 AI Credit Scoring & Loan Approval System")

st.markdown("---")

st.subheader("Enter Applicant Information")

# =========================
# INPUTS
# =========================

col1, col2 = st.columns(2)

with col1:

    status_account = st.number_input(
        "Status Account",
        min_value=0,
        max_value=10,
        value=1
    )

    month_duration = st.number_input(
        "Loan Duration (Months)",
        min_value=1,
        max_value=100,
        value=12
    )

    credit_history = st.number_input(
        "Credit History",
        min_value=0,
        max_value=10,
        value=1
    )

    purpose = st.number_input(
        "Purpose",
        min_value=0,
        max_value=20,
        value=1
    )

    credit_amount = st.number_input(
        "Credit Amount",
        min_value=0,
        value=5000
    )

    status_savings = st.number_input(
        "Savings Status",
        min_value=0,
        max_value=10,
        value=1
    )

    years_employment = st.number_input(
        "Years Employment",
        min_value=0,
        max_value=20,
        value=5
    )

    payment_to_income_ratio = st.number_input(
        "Payment To Income Ratio",
        min_value=0,
        max_value=10,
        value=2
    )

    status_and_sex = st.number_input(
        "Status And Sex",
        min_value=0,
        max_value=10,
        value=1
    )

    secondary_obligor = st.number_input(
        "Secondary Obligor",
        min_value=0,
        max_value=10,
        value=0
    )

with col2:

    residence_since = st.number_input(
        "Residence Since",
        min_value=0,
        max_value=20,
        value=2
    )

    collateral = st.number_input(
        "Collateral",
        min_value=0,
        max_value=20,
        value=1
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    other_installment_plans = st.number_input(
        "Other Installment Plans",
        min_value=0,
        max_value=20,
        value=0
    )

    housing = st.number_input(
    "Housing",
    min_value=0,
    max_value=5,
    value=1
    )

    n_credits = st.number_input(
        "Number Of Credits",
        min_value=0,
        max_value=20,
        value=1
    )

    job = st.number_input(
        "Job",
        min_value=0,
        max_value=20,
        value=1
    )

    n_guarantors = st.number_input(
        "Number Of Guarantors",
        min_value=0,
        max_value=20,
        value=0
    )

    telephone = st.number_input(
        "Telephone",
        min_value=0,
        max_value=5,
        value=1
    )

    is_foreign_worker = st.number_input(
        "Foreign Worker",
        min_value=0,
        max_value=5,
        value=1
    )

# =========================
# PREDICTION
# =========================

if st.button("🔍 Check Credit Score"):

    data = np.array([[
        status_account,
        month_duration,
        credit_history,
        purpose,
        credit_amount,
        status_savings,
        years_employment,
        payment_to_income_ratio,
        status_and_sex,
        secondary_obligor,
        residence_since,
        collateral,
        age,
        other_installment_plans,
        housing,
        n_credits,
        job,
        n_guarantors,
        telephone,
        is_foreign_worker
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)[0]

    probability = model.predict_proba(data_scaled)[0][1]

    # =====================
    # CREDIT SCORE
    # =====================

    credit_score = int(300 + (probability * 550))

    # =====================
    # RISK CATEGORY
    # =====================

    if credit_score >= 750:
        risk = "🟢 Low Risk"

    elif credit_score >= 650:
        risk = "🟡 Moderate Risk"

    else:
        risk = "🔴 High Risk"

    # =====================
    # LOAN DECISION
    # =====================

    if credit_score >= 700:
        decision = "✅ APPROVED"

    elif credit_score >= 600:
        decision = "🟠 MANUAL REVIEW"

    else:
        decision = "❌ REJECTED"

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Credit Score",
            credit_score
        )

    with c2:
        st.metric(
            "Default Probability",
            f"{(1-probability)*100:.2f}%"
        )

    with c3:
        st.metric(
            "Prediction",
            "Good Credit" if prediction == 1 else "Bad Credit"
        )

    st.success(f"Risk Level: {risk}")

    st.info(f"Loan Decision: {decision}")

    report = f"""
AI Credit Scoring Report

Credit Score: {credit_score}
Risk Level: {risk}
Loan Decision: {decision}
Default Probability: {(1-probability)*100:.2f}%
"""

    st.download_button(
        label="📥 Download Credit Report",
        data=report,
        file_name="credit_report.txt",
        mime="text/plain",
        key="download_btn"
    )
