# 📡 Telco Customer Churn — End-to-End ML Pipeline

> Predicting customer churn for a telecom provider using the IBM Telco dataset.  
> Full pipeline from raw data to a deployed interactive dashboard — built for production, not just notebooks.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-tuned-FF6600?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-8A2BE2?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 🎯 Business Problem

A telecom company loses **26.6% of its customers annually** — translating to **~$139K in lost monthly recurring revenue** (~$1.67M annualised). The goal of this project is to build a model that identifies at-risk customers *before* they leave, enabling the retention team to intervene proactively.

---

## 🗂️ Project Structure

```
telco-churn/
│
├── data/
│   ├── Telco_customer_churn.xlsx       # Raw IBM dataset
│   ├── telco_churn_clean.csv           # Output of cleaning notebook
│   └── telco_scored.csv                # Model predictions + risk tiers
│
├── notebooks/
│   ├── 01_cleaning.ipynb                   # Data cleaning & feature engineering
│   ├── 02_eda.ipynb                        # Exploratory data analysis
│   └── 03_modeling.ipynb                   # Model training, tuning & explainability
│
├── dashboard.py                        # Streamlit analytics dashboard
└── README.md
```

---

## 🔬 Pipeline Overview

### `01_cleaning.ipynb` — Data Cleaning

| Step | Issue | Decision |
|------|-------|----------|
| `Total Charges` dtype | Stored as `object` due to 11 whitespace rows | Coerced to `float64`; 11 zero-tenure rows dropped (0.15%) |
| `Churn Reason` nulls | 5,174 nulls (~73%) | Structural nulls — only churned customers fill this field |
| Zero-variance columns | `Count`, `Country`, `State` | Dropped — no predictive signal |
| Geographic columns | `City`, `Zip Code`, lat/lon | Dropped — too high cardinality without engineering |
| Data leakage | `Churn Score`, `CLTV`, `Churn Label`, `Churn Reason` | Dropped — post-event or circular features |
| `CustomerID` | Row identifier | Saved separately for prediction mapping; not used as a feature |
| Binary encoding | Yes/No columns | Encoded here (1/0); multi-class deferred to modeling pipeline |
| Feature engineering | — | `charges_per_month`, `addon_count` |

**Result:** 7,043 → 7,010 rows · 33 → 22 columns · 0 nulls

---

### `02_eda.ipynb` — Exploratory Data Analysis

Key findings that directly shaped modeling decisions:

**Finding 1 — Contract type is the single biggest lever**
Month-to-month customers churn at **~43%** — roughly 15× the rate of two-year contract customers (~3%). Converting even 10% of month-to-month customers to annual contracts is the highest-leverage retention action available.

**Finding 2 — Churn is front-loaded**
Customers in their first year churn at **~47%**; by years 4–6 this drops to **~10%**. Retention spend should be concentrated in the first 90 days.

**Finding 3 — Price sensitivity is real**
Churned customers pay **~$13/month more** on average than retained customers ($74 vs $61), but their total charges are far lower — they leave before accumulating value.

**Finding 4 — Payment method signals intent**
Electronic check customers churn at significantly higher rates than those on automatic payment plans — likely a proxy for lower engagement and commitment.

> Charts saved to `data/`: `eda_01_contract_churn.png`, `eda_02_tenure_churn.png`, `eda_03_charges_churn.png`, etc.

---

### `03_modeling.ipynb` — Modeling

**Preprocessing**
- `OneHotEncoder` for `contract`, `internet_service`, `payment_method` — fitted **inside a Pipeline after the train/test split** to prevent leakage
- `StandardScaler` for logistic regression baseline; passthrough for XGBoost
- Stratified 80/20 split to preserve the 26.6% churn rate in both sets

**Models evaluated**

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression (baseline) | ~0.854 | — |
| XGBoost (default) | ~0.85 | — |
| **XGBoost (tuned)** | **~0.858** | **best** |

- `scale_pos_weight = neg/pos` to handle class imbalance without resampling
- `RandomizedSearchCV` (n_iter=40, 5-fold CV) for hyperparameter tuning
- `early_stopping_rounds=30` to prevent overfitting
- Business threshold set at **0.40** — catching churners (recall) is more valuable than avoiding false positives in this context

**Explainability**
SHAP values confirm EDA predictions: `contract`, `tenure_months`, and `monthly_charges` are consistently top features. Each customer prediction is fully explainable.

**Output:** `data/model_xgb.pkl`, `data/telco_scored.csv` (churn probability + risk tier per customer)

---

## 🖥️ Dashboard

An interactive Streamlit dashboard with four tabs:

| Tab | Content |
|-----|---------|
| **Overview** | Churn distribution, contract breakdown, tenure & charges distributions |
| **Segment Analysis** | Churn by internet service, payment method, demographics, scatter analysis |
| **ML Model** | Accuracy, AUC-ROC, recall, feature importance, confusion matrix |
| **Customer Prediction** | Live churn risk score for a configurable customer profile + retention recommendations |

**KPIs displayed:** total customers, churn rate, lost customers, monthly revenue, revenue lost to churn — all responsive to sidebar filters (contract type, internet service, tenure range, monthly charges).

The dashboard loads the tuned XGBoost model from `data/model_xgb.pkl` if present, and falls back to a Random Forest trained on the fly otherwise.

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/your-username/telco-churn.git
cd telco-churn

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook 01_cleaning.ipynb
jupyter notebook 02_eda.ipynb
jupyter notebook 03_modeling.ipynb

# 4. Launch the dashboard
streamlit run dashboard.py
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
streamlit
plotly
openpyxl
```

> Full pinned versions in `requirements.txt`

---

## 📊 Dataset

**IBM Telco Customer Churn** — 7,043 customers × 33 features  
Available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) or the [IBM GitHub repository](https://github.com/IBM/telco-customer-churn-on-icp4d).

Features include demographics, account information, subscribed services, contract type, payment method, and monthly/total charges. Target variable: `Churn` (Yes/No).

---

## 💡 Key Design Decisions

**Why XGBoost over logistic regression?**
XGBoost handles correlated features (`monthly_charges` / `charges_per_month`, r = 0.996) gracefully, supports `scale_pos_weight` for class imbalance natively, and has built-in SHAP support. It meaningfully outperforms the logistic baseline on both ROC-AUC and PR-AUC.

**Why threshold 0.40 and not 0.50?**
In a churn context, the cost of a false negative (missing a churner who then leaves) is higher than the cost of a false positive (offering a retention incentive to someone who wouldn't have left). Lowering the threshold increases recall at an acceptable precision trade-off.

**Why encode binary features in cleaning but multi-class in the pipeline?**
Binary Yes/No encoding is deterministic and has no fit step — it can safely happen before the split. Multi-class OHE has a fit step (learning the category vocabulary), so it must happen inside the pipeline after splitting to avoid data leakage.

---

## 📁 Outputs

| File | Description |
|------|-------------|
| `data/telco_churn_clean.csv` | Clean, model-ready dataset |
| `data/telco_scored.csv` | All customers with `churn_prob` and `risk_tier` |
| `data/model_xgb.pkl` | Serialised tuned XGBoost model + preprocessor |
| `data/eda_*.png` | EDA charts (6 files) |

---

## 👤 Author : Hajar AITLAKSSIR

Built as a portfolio project demonstrating end-to-end ML engineering — from messy raw data to a production-ready dashboard with explainable predictions.
