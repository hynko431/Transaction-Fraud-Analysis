# **Data Set** :-
## https://drive.usercontent.google.com/download?id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV&export=download&authuser=0 

# Transaction-Fraud-Analysis
"Detect fraud before it strikes 🚨 | A complete machine learning pipeline for transaction fraud detection using Random Forests, data balancing with SMOTE, Optuna-based optimization, and real-time prediction support."


# 🛡️ Fraud Detection using Machine Learning

This project focuses on detecting fraudulent financial transactions using a machine learning pipeline. It leverages data preprocessing, SMOTE for class balancing, a Random Forest classifier, and Optuna for hyperparameter tuning. The system is evaluated using ROC AUC and applied to real-like transaction samples.

---

## 📂 Dataset

- The dataset consists of **6,362,621 rows** and **11 columns**.
- Each transaction contains fields such as `amount`, `oldbalanceOrg`, `newbalanceOrig`, `type`, and the target variable `isFraud`.

---

## ⚙️ Features Used

- Transaction details: `amount`, `oldbalanceOrg`, `newbalanceOrig`, etc.
- Encoded categorical feature: `type` (e.g., TRANSFER, PAYMENT, CASH_OUT)
- Derived feature: `merchantFlag` (optional)
- Target: `isFraud`

---

## 🧠 Model & Pipeline

- **Model**: Random Forest Classifier
- **Balancing**: SMOTE to handle class imbalance
- **Tuning**: Optuna used for hyperparameter optimization
- **Evaluation Metric**: ROC AUC Score

---

## 📈 Model Performance

| Model Variant                | ROC AUC Score |
|-----------------------------|---------------|
| Baseline Random Forest      | 0.97+         |
| Tuned Random Forest (Optuna)| 0.98+         |

> ✅ Results may vary slightly based on cross-validation and random seeds.

---

## 🧪 Inference Demo

You can test the model using custom samples or by uploading a CSV (`fraud_detection_samples_raw.csv`) containing new transaction data. Sample predictions include fraud label and its associated probability.

```python
# Predict on sample CSV
df_samples = pd.read_csv("fraud_detection_samples.csv")
results = predict_from_csv(rf_model_sm, df_samples, features)
print(results[['amount', 'prediction', 'probability']])

## Installation:-
git clone https://github.com/hynko431/Transaction-Fraud-Analysis.git
cd Transaction-Fraud-Analysis

## Key Libraries Used
- scikit-learn
- pandas, numpy
- optuna
- imbalanced-learn

