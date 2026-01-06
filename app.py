# loan_approval_app_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page config (must be first)
# -------------------------------
st.set_page_config(
    page_title="Loan Eligibility & Credit Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Model & Scaler
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("loan_approval_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

log_reg, scaler = load_model()

# -------------------------------
# Helper Functions
# -------------------------------
def risk_band(pa):
    if pa >= 0.70:
        return 'Low Risk'
    elif pa >= 0.40:
        return 'Medium Risk'
    else:
        return 'High Risk'

def credit_decision(risk):
    if risk == 'Low Risk':
        return 'Approve'
    elif risk == 'Medium Risk':
        return 'Review'
    else:
        return 'Reject'

def credit_score(pa):
    return round(300 + pa * 550, 0)

def score_grade(score):
    if score >= 750:
        return 'A'
    elif score >= 700:
        return 'B'
    elif score >= 650:
        return 'C'
    elif score >= 600:
        return 'D'
    else:
        return 'E'

def score_decision(grade):
    if grade in ['A', 'B']:
        return 'Approve'
    elif grade == 'C':
        return 'Review'
    else:
        return 'Reject'

def preprocess_input(input_df, scaler):
    # Drop irrelevant columns if they exist
    input_df = input_df.drop(columns=['Customer_ID', 'Loan_Status'], errors='ignore')

    # Numerical columns
    num_cols = ['Dependents', 'Applicant_Income', 'Coapplicant_Income', 'Loan_Amount',
                'Loan_Amount_Term', 'Credit_History', 'Total_Income', 'Loan_to_Income']
    
    # Create new features
    input_df['Total_Income'] = input_df['Applicant_Income'] + input_df['Coapplicant_Income']
    input_df['Loan_to_Income'] = input_df['Loan_Amount'] / input_df['Total_Income']
    
    # Categorical columns
    cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    input_df_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True, dtype=int)
    
    # Ensure all expected dummy columns exist
    expected_cols = ['Gender_Male', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes',
                     'Property_Area_Semiurban', 'Property_Area_Urban']
    for col in expected_cols:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    # Scale numerical features
    X_num_scaled = scaler.transform(input_df_encoded[num_cols])
    X_final = pd.concat([pd.DataFrame(X_num_scaled, columns=num_cols, index=input_df_encoded.index),
                         input_df_encoded.drop(columns=num_cols)], axis=1)
    return X_final


def make_predictions(input_df):
    X = preprocess_input(input_df, scaler)
    pa = log_reg.predict_proba(X)[:, 1]
    risk = [risk_band(p) for p in pa]
    decision = [credit_decision(r) for r in risk]
    score = [credit_score(p) for p in pa]
    grade = [score_grade(s) for s in score]
    score_dec = [score_decision(g) for g in grade]
    
    results = input_df.copy()
    results['Predicted_Approval_Probability'] = pa
    results['Risk_Level'] = risk
    results['Credit_Decision'] = decision
    results['Credit_Score'] = score
    results['Score_Grade'] = grade
    results['Score_Decision'] = score_dec
    return results

# -------------------------------
# Streamlit Layout
# -------------------------------
st.title("Loan Eligibility & Credit Scoring App")
st.markdown("Predict loan approval, assess risk, calculate credit scores, and visualize results.")

mode = st.radio("Select Mode", ["Single Applicant", "Batch Upload (CSV)"])

# -------------------------------
# Single Applicant Mode
# -------------------------------
if mode == "Single Applicant":
    st.sidebar.header("Applicant Information")
    
    def user_input_features():
        gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
        married = st.sidebar.selectbox("Married", ["Yes", "No"])
        dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 0)
        education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.sidebar.number_input("Applicant Income", 0, 1000000, 5000)
        coapplicant_income = st.sidebar.number_input("Coapplicant Income", 0, 1000000, 0)
        loan_amount = st.sidebar.number_input("Loan Amount", 0, 10000, 200)
        loan_term = st.sidebar.number_input("Loan Amount Term (months)", 12, 480, 360)
        credit_history = st.sidebar.selectbox("Credit History (1=Good, 0=Bad)", [1, 0])
        property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'Applicant_Income': applicant_income,
            'Coapplicant_Income': coapplicant_income,
            'Loan_Amount': loan_amount,
            'Loan_Amount_Term': loan_term,
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()
    results = make_predictions(input_df)
    
    st.subheader("Prediction Results")
    st.dataframe(results)

# -------------------------------
# Batch Upload Mode
# -------------------------------
else:
    st.subheader("Upload CSV file for multiple applicants")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(batch_df)
        
        results = make_predictions(batch_df)
        st.subheader("Prediction Results")
        st.dataframe(results)

        # -------------------------------
        # Visualizations using Matplotlib
        # -------------------------------
        st.subheader("Visualizations")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Risk Level Distribution
        risk_counts = results['Risk_Level'].value_counts()
        axes[0].bar(risk_counts.index, risk_counts.values, color=['green', 'orange', 'red'])
        axes[0].set_title("Risk Level Distribution")
        axes[0].set_ylabel("Count")

        # Credit Decision Distribution
        decision_counts = results['Credit_Decision'].value_counts()
        axes[1].bar(decision_counts.index, decision_counts.values, color=['green', 'orange', 'red'])
        axes[1].set_title("Credit Decision Distribution")
        axes[1].set_ylabel("Count")

        # Credit Score Distribution
        axes[2].hist(results['Credit_Score'], bins=10, color='skyblue', edgecolor='black')
        axes[2].set_title("Credit Score Distribution")
        axes[2].set_xlabel("Credit Score")
        axes[2].set_ylabel("Count")

        st.pyplot(fig)

        # -------------------------------
        # Download CSV
        # -------------------------------
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='loan_approval_predictions.csv',
            mime='text/csv'
        )
