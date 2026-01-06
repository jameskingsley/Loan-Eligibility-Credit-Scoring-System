# Loan-Eligibility-Credit-Scoring-System


A complete end-to-end credit risk scoring system built using Python, Machine Learning, and Streamlit, designed to simulate real-world loan underwriting and decision-making processes used in banks and fintech institutions.

This project covers data preprocessing, feature engineering, model development, risk segmentation, credit scorecard creation, and deployment for both real-time and batch loan assessments.

###### Project Overview

Financial institutions must evaluate loan applicants by balancing risk, profitability, and fairness.
This project implements a Logistic Regression–based credit scoring model that:

Predicts loan approval probability

Segments applicants into risk bands

Assigns standardized credit scores (300–850)

Supports real-time and batch decisioning

Is fully deployable via a Streamlit web application

###### Objectives

Build an interpretable baseline credit risk model

Engineer business-relevant financial features

Implement risk-based decision policies

Create a credit scorecard aligned with industry standards

Deploy a user-friendly application for operational use



###### Tech Stack

Programming Language: Python

Libraries:

pandas, numpy

scikit-learn

matplotlib

joblib

Model: Logistic Regression

Deployment: Streamlit

###### Feature Engineering

The project applies domain-driven transformations to reflect real underwriting logic:

Total Income = Applicant Income + Co-applicant Income

Loan-to-Income Ratio = Loan Amount / Total Income

Categorical encoding using one-hot encoding

Numerical scaling using StandardScaler

Only numerical variables are scaled to preserve the interpretability of binary indicators.

###### Model Development

Algorithm: Logistic Regression (baseline credit model)

Reason for choice:

Interpretable coefficients

Probability-based output

Industry-accepted baseline model

###### Evaluation Metrics:

Accuracy

Precision & Recall

ROC-AUC (~0.80)


 ###### Credit Scorecard (300–850)

A standardized credit score is generated using:

Credit Score = 300 + (Approval Probability × 550)


###### Single Applicant Mode

Manual input of applicant details

Instant credit decision

Risk level, probability, score & grade display

###### Batch Prediction Mode

Upload CSV file of applicants

Automated scoring and decisioning

Downloadable prediction results

###### Saved artifacts:

loan_approval_model.pkl

scaler.pkl

###### Key Takeaways

Demonstrates real-world credit risk modeling

Emphasizes interpretability and business logic

Follows best practices in preprocessing & deployment

Suitable for banking, fintech, and risk analytics portfolios

###### Author

James Kingsley Philip
Data Scientist | Machine Learning Engineer
