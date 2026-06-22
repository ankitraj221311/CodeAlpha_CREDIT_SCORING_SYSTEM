# AI Credit Scoring & Loan Approval System

## Project Overview

The AI Credit Scoring & Loan Approval System is a Machine Learning project designed to predict an individual's creditworthiness based on historical financial data. The system evaluates loan applicants and generates a credit score, risk category, default probability, and loan approval decision.

The project uses data preprocessing, feature engineering, class balancing, model training, and an interactive Streamlit dashboard to provide real-time credit risk predictions.

---

## Objectives

* Predict customer creditworthiness.
* Estimate the probability of loan default.
* Generate a credit score ranging from 300–850.
* Categorize applicants into risk levels.
* Automate loan approval decisions.
* Provide an interactive web-based dashboard.

---

## 📊 Dataset

Dataset Used: German Credit Dataset

Features include:

* Status Account
* Credit History
* Loan Purpose
* Credit Amount
* Savings Status
* Employment Duration
* Payment to Income Ratio
* Age
* Housing Status
* Number of Credits
* Job Type
* Telephone Status
* Foreign Worker Status
* And other financial indicators

Target Variable:

* 1 → Good Credit
* 0 → Bad Credit

---

## 🛠 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-Learn
* Imbalanced-Learn (SMOTE)
* Joblib
* Streamlit

### Machine Learning Algorithm

* Random Forest Classifier

---

## ⚙️ Project Workflow

### 1. Data Preprocessing

* Missing value handling
* Categorical feature encoding using LabelEncoder
* Feature selection

### 2. Data Splitting

* Train-Test Split (80:20)

### 3. Feature Scaling

* StandardScaler

### 4. Class Balancing

* SMOTE (Synthetic Minority Oversampling Technique)

### 5. Model Training

* Random Forest Classifier

### 6. Model Evaluation

Metrics used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

### 7. Model Deployment

* Streamlit Web Application

---

## 📈 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 78.5% |
| Precision | 83.1% |
| Recall    | 87.2% |
| F1 Score  | 85.1% |
| ROC-AUC   | 82.3% |

---

## Features

### Credit Risk Prediction

Predicts whether a customer has Good Credit or Bad Credit.

### Credit Score Generation

Generates a score between 300 and 850.

### Risk Assessment

Classifies applicants as:

* 🟢 Low Risk
* 🟡 Moderate Risk
* 🔴 High Risk

### Loan Decision System

Provides automated decisions:

* ✅ Approved
* 🟠 Manual Review
* ❌ Rejected

### Interactive Dashboard

Built using Streamlit for real-time predictions.

---

## Project Structure

```text
Credit_Scoring_System/
│
├── datas/
│   └── german_credit_data.csv
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
│
├── app.py
├── train.py
├── preprocess.py
├── requirements.txt
└── README.md
```

## ▶️ Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd Credit_Scoring_System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

This will generate:

* model.pkl
* scaler.pkl
* encoders.pkl

inside the models folder.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open your browser and visit:

```text
http://localhost:8501
```

---

## Future Enhancements

* XGBoost Integration
* SHAP Explainability
* PDF Credit Report Generation
* Cloud Deployment
* Real-Time Database Integration
* Advanced Credit Score Analytics

---

## Author

Ankit Raj Singh

Machine Learning & AI Enthusiast

---

## If you found this project useful, please give it a star on GitHub.
