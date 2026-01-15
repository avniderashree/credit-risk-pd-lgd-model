# Credit Risk PD/LGD Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive credit risk modeling framework implementing **Probability of Default (PD)** and **Loss Given Default (LGD)** models using machine learning, with full model validation and Expected Loss calculation.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [What You'll See](#-what-youll-see)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Sample Results](#-sample-results)
- [Usage Examples](#-usage-examples)
- [Model Validation](#-model-validation)
- [Visualizations](#-visualizations)
- [Technical Skills Demonstrated](#-technical-skills-demonstrated)
- [References](#-references)

---

## 📊 Overview

Credit risk modeling is fundamental to banking and lending. This project implements the three pillars of credit risk:

| Component | Definition | Model Type |
|-----------|------------|------------|
| **PD** | Probability of Default | Classification |
| **LGD** | Loss Given Default | Regression |
| **EAD** | Exposure at Default | Given/Calculated |

**Expected Loss (EL)** = PD × LGD × EAD

### Real-World Applications

- **Loan Approval**: Score applicants for default risk
- **Pricing**: Risk-based interest rate setting
- **Provisioning**: IFRS 9 / CECL expected credit loss
- **Capital Requirements**: Basel III/IV regulatory capital

---

## ✨ Features

### PD Models Implemented

| Model | Algorithm | Key Advantage |
|-------|-----------|---------------|
| **Logistic Regression** | Linear | Interpretable, regulatory-friendly |
| **Random Forest** | Ensemble | Handles non-linearity |
| **XGBoost** | Gradient Boosting | Best predictive performance |

### LGD Models

| Model | Algorithm |
|-------|-----------|
| **Ridge Regression** | Linear with regularization |
| **Random Forest** | Non-linear regression |
| **XGBoost** | Gradient boosting regression |

### Validation Metrics

- ✅ **ROC-AUC** — Discrimination power
- ✅ **Gini Coefficient** — 2×AUC - 1
- ✅ **KS Statistic** — Maximum class separation
- ✅ **Calibration Curve** — Predicted vs actual rates
- ✅ **Lift Chart** — Model effectiveness by decile
- ✅ **PSI** — Population Stability Index for monitoring

### Additional Features

- ✅ **Information Value (IV)** — Feature predictive power
- ✅ **Weight of Evidence (WoE)** — Feature transformation
- ✅ **Risk Segmentation** — Bucket loans by risk level
- ✅ **Expected Loss Calculation** — Full EL pipeline

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/avniderashree/credit-risk-pd-lgd-model.git
cd credit-risk-pd-lgd-model
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Analysis

```bash
python main.py
```

### Step 5: Explore Interactively

```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

---

## 🖥️ What You'll See

Running `python main.py` produces:

```
============================================================
 CREDIT RISK PD/LGD MODEL
============================================================

This analysis builds credit risk models for:
  1. Probability of Default (PD) - Classification
  2. Loss Given Default (LGD) - Regression
  3. Expected Loss (EL = PD × LGD × EAD)

------------------------------------------------------------
 STEP 1: Loading Credit Data
------------------------------------------------------------
Generated 5000 synthetic credit records
Default rate: 15.32%

Dataset Statistics:
  • Training samples: 4000
  • Test samples: 1000
  • Features: 15
  • Default rate (train): 15.28%

------------------------------------------------------------
 STEP 2: Feature Engineering
------------------------------------------------------------

Top Features by Information Value:
  • credit_score             IV=0.4521 (Strong)
  • delinquencies_2yr        IV=0.3812 (Strong)
  • dti_ratio                IV=0.2156 (Medium)
  • credit_utilization       IV=0.1823 (Medium)

------------------------------------------------------------
 STEP 3: Training PD Models
------------------------------------------------------------

PD Model Comparison:
------------------------------------------------------------
          Model   ROC-AUC     Gini   KS Statistic
        XGBoost    0.8234   0.6468        0.5012
  Random Forest    0.8156   0.6312        0.4823
     Logistic      0.7845   0.5690        0.4234

✓ Best PD Model: XGBoost
  • ROC-AUC: 0.8234
  • Gini: 0.6468
  • KS Statistic: 0.5012

------------------------------------------------------------
 STEP 5: Expected Loss Calculation
------------------------------------------------------------

Expected Loss Statistics:
  • Average PD:  14.82%
  • Average LGD: 42.15%
  • Total Expected Loss: $2,847,234

📊 Risk Segmentation:
           Count  Avg PD   Total EL
Low          523   2.3%   $142,345
Medium       287   9.8%   $567,234
High         134  22.4%   $987,654
Very High     56  45.2%  $1,150,001

------------------------------------------------------------
 STEP 6: Generating Visualizations
------------------------------------------------------------

Saving charts to ./output/ directory...
  ✓ roc_curve.png
  ✓ ks_chart.png
  ✓ calibration_curve.png
  ✓ feature_importance.png
  ✓ lift_chart.png
  ✓ model_comparison.png
  ✓ probability_distribution.png

============================================================
 ANALYSIS COMPLETE
============================================================

📁 Output files saved to ./output/
📁 Models saved to ./models/

Done! ✅
```

---

## 📁 Project Structure

```
credit-risk-pd-lgd-model/
│
├── main.py                     # Main execution script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── src/                        # Core modules
│   ├── data_loader.py          # Data loading & synthetic generation
│   ├── feature_engineering.py  # WoE, IV, derived features
│   ├── pd_model.py             # PD classification models
│   ├── lgd_model.py            # LGD regression models
│   ├── evaluation.py           # Gini, KS, PSI metrics
│   └── visualization.py        # Charts and plots
│
├── notebooks/
│   └── credit_risk_analysis.ipynb
│
├── tests/
│   └── test_credit_risk.py
│
├── models/                     # Saved trained models
│   ├── pd_model.pkl
│   ├── lgd_model.pkl
│   └── feature_engineer.pkl
│
└── output/                     # Generated visualizations
    ├── roc_curve.png
    ├── ks_chart.png
    ├── calibration_curve.png
    ├── feature_importance.png
    ├── lift_chart.png
    ├── model_comparison.png
    └── probability_distribution.png
```

---

## 🧮 Methodology

### PD Modeling

**Objective**: Predict the probability that a borrower will default.

**Target Variable**: Binary (0 = No Default, 1 = Default)

**Features Used**:
- Credit score, DTI ratio, loan amount
- Employment length, annual income
- Delinquencies, bankruptcies
- Credit utilization, number of credit lines

**Model Selection**: XGBoost typically wins due to:
- Handling of non-linear relationships
- Built-in regularization
- Native handling of missing values

### LGD Modeling

**Objective**: Predict the loss percentage if default occurs.

**Target Variable**: Continuous [0, 1]

**Trained on**: Defaulted loans only

**Typical LGD**: 40-50% for unsecured consumer loans

### Expected Loss

```
EL = PD × LGD × EAD

Where:
- PD = Probability of default (0-1)
- LGD = Loss given default (0-1)
- EAD = Exposure at default ($)
```

---

## 📈 Sample Results

### PD Model Performance

| Model | ROC-AUC | Gini | KS Statistic |
|-------|---------|------|--------------|
| XGBoost | 0.8234 | 0.6468 | 0.5012 |
| Random Forest | 0.8156 | 0.6312 | 0.4823 |
| Logistic | 0.7845 | 0.5690 | 0.4234 |

### LGD Model Performance

| Model | RMSE | MAE | Mean LGD |
|-------|------|-----|----------|
| XGBoost | 0.1523 | 0.1102 | 42.15% |
| Random Forest | 0.1612 | 0.1198 | 41.87% |
| Linear | 0.1845 | 0.1423 | 40.23% |

### Key Insights

1. **XGBoost dominates** both PD and LGD tasks
2. **Credit score** is the strongest predictor (IV=0.45)
3. **High KS (0.50)** indicates strong class separation
4. **Calibration is good** — predicted probabilities match actual rates

---

## 💻 Usage Examples

### Basic PD Prediction

```python
from src.data_loader import prepare_credit_data
from src.feature_engineering import FeatureEngineer
from src.pd_model import train_xgboost

# Load data
X_train, X_test, y_train, y_test = prepare_credit_data()

# Engineer features
fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train)
X_test_fe = fe.transform(X_test)

# Train model
result = train_xgboost(X_train_fe, y_train['default'], 
                       X_test_fe, y_test['default'])

print(f"ROC-AUC: {result.roc_auc:.4f}")
print(f"Gini: {result.gini:.4f}")
```

### Score New Applications

```python
import joblib

# Load saved model
model = joblib.load('models/pd_model.pkl')
fe = joblib.load('models/feature_engineer.pkl')

# New application
new_app = pd.DataFrame({
    'credit_score': [720],
    'loan_amount': [15000],
    'dti_ratio': [25],
    ...
})

# Predict PD
X_new = fe.transform(new_app)
pd_score = model.predict_proba(X_new)[:, 1]
print(f"Default Probability: {pd_score[0]:.2%}")
```

---

## 🔍 Model Validation

### Discrimination Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **ROC-AUC** | Area under ROC | 0.5 = random, 1.0 = perfect |
| **Gini** | 2×AUC - 1 | 0 = random, 1.0 = perfect |
| **KS** | max(TPR - FPR) | Max separation between classes |

### Calibration

- **Hosmer-Lemeshow Test**: Tests if predicted = actual by decile
- **Calibration Curve**: Visual comparison

### Stability

- **PSI < 0.10**: No significant shift
- **PSI 0.10-0.25**: Moderate shift, monitor
- **PSI > 0.25**: Significant shift, retrain

---

## 📊 Visualizations

The project generates 7 professional charts:

1. **ROC Curve** — AUC and Gini visualization
2. **KS Chart** — Cumulative distribution separation
3. **Calibration Curve** — Predicted vs actual rates
4. **Feature Importance** — Top predictive features
5. **Lift Chart** — Model effectiveness by decile
6. **Model Comparison** — Compare all models
7. **Probability Distribution** — Score separation by class

---

## 🎓 Technical Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Credit Risk** | PD, LGD, EAD, Expected Loss, Basel framework |
| **Machine Learning** | XGBoost, Random Forest, Logistic Regression |
| **Feature Engineering** | WoE, IV, derived features, scaling |
| **Model Validation** | ROC, Gini, KS, calibration, PSI |
| **Python** | scikit-learn, xgboost, pandas, numpy |
| **Software Eng** | Modular design, unit testing, model persistence |

---

## 📚 References

1. Siddiqi, N. (2012). *Credit Risk Scorecards*. Wiley.
2. Basel Committee. *Basel III: A Global Regulatory Framework*.
3. Thomas, L.C. (2009). *Consumer Credit Models*. Oxford.
4. IFRS 9. *Financial Instruments - Expected Credit Loss*.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 👤 Author

**Avni Derashree**  
Quantitative Risk Analyst | Python | Machine Learning

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/avniderashree/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/avniderashree)

---

*Part of a quantitative finance portfolio. See also:*
- [Portfolio VaR Calculator](https://github.com/avniderashree/portfolio-var-calculator)
- [GARCH Volatility Forecaster](https://github.com/avniderashree/garch-volatility-forecaster)
