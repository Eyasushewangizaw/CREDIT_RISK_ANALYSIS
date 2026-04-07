# CREDIT_RISK_ANALYSIS



A machine learning project that predicts whether a loan applicant will default, built using the German Credit Risk dataset. This project mirrors real-world bank data science workflows — from raw data to explainable model outputs.

---

## Project Overview

Banks lose billions every year from loan defaults. This project builds a classification model that predicts credit default risk from applicant data, using techniques directly applicable to roles in banking and financial data science.

**Dataset:** German Credit Risk (UCI Machine Learning Repository) — 1,000 loan applicants, 20 features, binary target (good/bad credit)

**Key result:** Logistic Regression AUC of 0.817, meaning the model correctly ranks 81.7% of good vs bad credit applicants.

---

## Project Structure

```
credit-project/
│
├── german.data              # Raw dataset
├── 01_load_data.py          # Load and preview the dataset
├── 02_eda.py                # Exploratory data analysis + visualizations
├── 03_prepare_data.py       # Data cleaning and feature encoding
├── 04_model.py              # Logistic regression baseline model
├── 05_xgboost.py            # XGBoost model with feature importance
├── 06_shap.py               # SHAP explainability chart
├── default_rate.png         # Chart: good vs bad credit distribution
└── shap_summary.png         # Chart: SHAP feature importance
```

---

## Key Findings

**From exploratory analysis:**
- 70% of applicants have good credit, 30% bad — a class imbalance common in real banking data
- Applicants who defaulted borrowed 32% more on average (3,938 vs 2,985)
- Defaulters were younger on average (33.9 vs 36.2 years)
- Defaulted loans lasted longer on average (24.8 vs 19.2 months)

**From SHAP explainability:**
- Checking account balance is the strongest predictor of default
- Higher loan amounts and longer durations increase default risk
- Poor credit history and younger age are significant risk factors

---

## Model Results

| Model | Accuracy | AUC Score |
|---|---|---|
| Logistic Regression | 78% | 0.817 |
| XGBoost | 79% | 0.799 |

Logistic regression outperformed XGBoost on this dataset — expected behavior on smaller datasets (1,000 rows). XGBoost advantages appear at larger scale. Both models were evaluated on a held-out test set of 200 applicants.

---

## Tools and Libraries

- **Python** — core language
- **pandas** — data loading and manipulation
- **scikit-learn** — logistic regression, preprocessing, evaluation metrics
- **XGBoost** — gradient boosting model
- **SHAP** — model explainability
- **matplotlib** — visualizations
- **VS Code** — development environment

---

## Skills Demonstrated

- Exploratory data analysis and statistical summarization
- Feature engineering and categorical encoding
- Binary classification modeling
- Model evaluation with AUC, precision, recall, F1
- Model explainability with SHAP values
- End-to-end ML pipeline from raw data to insights

---

## Dataset Source

German Credit Risk dataset from the UCI Machine Learning Repository.
Original source: Professor Dr. Hans Hofmann, Universität Hamburg.
https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
