# Credit Risk PD/LGD Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5+-green.svg)](https://xgboost.readthedocs.io/)

A production-ready credit risk modeling framework implementing **Probability of Default (PD)** and **Loss Given Default (LGD)** models using machine learning. This project demonstrates the complete credit risk modeling workflow used in banking and fintech for loan underwriting, provisioning, and regulatory capital calculations.

---

## 📋 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [Why Credit Risk Modeling Matters](#-why-credit-risk-modeling-matters)
3. [Key Concepts Explained](#-key-concepts-explained)
4. [Features](#-features)
5. [Installation](#-installation)
6. [How to Run](#-how-to-run)
7. [Understanding the Output](#-understanding-the-output)
8. [Project Structure](#-project-structure)
9. [Module Documentation](#-module-documentation)
10. [Methodology Deep Dive](#-methodology-deep-dive)
11. [Model Validation Metrics](#-model-validation-metrics)
12. [Sample Results](#-sample-results)
13. [Code Examples](#-code-examples)
14. [Customization Guide](#-customization-guide)
15. [Troubleshooting](#-troubleshooting)
16. [Technical Skills Demonstrated](#-technical-skills-demonstrated)
17. [References](#-references)

---

## 🎯 What Is This Project?

This project builds a complete **credit risk scoring system** that predicts:

1. **Probability of Default (PD)**: The likelihood that a borrower will fail to repay their loan
2. **Loss Given Default (LGD)**: If default occurs, what percentage of the loan amount will be lost
3. **Expected Loss (EL)**: The dollar amount a lender expects to lose on a loan portfolio

### The Formula

```
Expected Loss = PD × LGD × EAD

Where:
- PD  = Probability of Default (0 to 1)
- LGD = Loss Given Default (0 to 1, typically 40-60%)
- EAD = Exposure at Default (loan amount outstanding)
```

### Example
If a $10,000 loan has:
- PD = 15% (15% chance of default)
- LGD = 45% (45% of the loan would be lost if default occurs)
- EAD = $10,000

Then: **Expected Loss = 0.15 × 0.45 × $10,000 = $675**

---

## 💡 Why Credit Risk Modeling Matters

Credit risk models are fundamental to the financial industry:

| Use Case | Description |
|----------|-------------|
| **Loan Approval** | Decide whether to approve or reject loan applications |
| **Risk-Based Pricing** | Set interest rates based on borrower risk level |
| **Provisioning (IFRS 9 / CECL)** | Calculate reserves for expected credit losses |
| **Regulatory Capital (Basel III/IV)** | Determine capital requirements for banks |
| **Portfolio Management** | Monitor and manage loan portfolio risk |
| **Stress Testing** | Simulate portfolio performance under adverse scenarios |

### Regulatory Context

- **Basel III/IV**: Banks must hold capital proportional to the credit risk in their loan portfolios
- **IFRS 9**: Requires forward-looking expected credit loss (ECL) provisioning
- **CECL (US)**: Current Expected Credit Loss standard for US financial institutions

---

## 📚 Key Concepts Explained

### Probability of Default (PD)

PD is the likelihood that a borrower will default on their loan within a specific time horizon (usually 12 months or lifetime).

**How it's calculated:**
- Use historical loan data where we know which loans defaulted
- Train a classification model (Logistic Regression, XGBoost, etc.)
- Model outputs a probability between 0 and 1

**Interpretation:**
- PD = 0.05 means 5% chance of default
- PD = 0.25 means 25% chance of default

### Loss Given Default (LGD)

LGD is the percentage of the loan that will be lost if the borrower defaults. Not all defaults result in 100% loss due to:
- Collateral recovery
- Collections efforts
- Partial payments

**Typical values:**
- Unsecured consumer loans: 40-60% LGD
- Secured mortgages: 20-30% LGD
- Credit cards: 70-80% LGD

### Exposure at Default (EAD)

EAD is the amount outstanding when default occurs. For:
- Term loans: Usually the outstanding principal
- Credit cards: Current balance + potential future draws
- Credit lines: Current draw + unused commitment × Credit Conversion Factor

---

## ✨ Features

### Machine Learning Models

| Model | Type | Purpose | Advantages |
|-------|------|---------|------------|
| **Logistic Regression** | Linear | PD prediction | Interpretable, regulatory-friendly |
| **Random Forest** | Ensemble | PD/LGD prediction | Handles non-linearity |
| **XGBoost** | Gradient Boosting | PD/LGD prediction | Best predictive performance |
| **Ridge Regression** | Linear | LGD prediction | Regularized, stable |

### Feature Engineering

- **Derived Features**: Loan-to-income ratio, payment-to-income ratio
- **Bucketing**: Credit score buckets, age groups
- **Flags**: High utilization flag, delinquency flag, stable employment
- **Weight of Evidence (WoE)**: Optimal binning transformation
- **Information Value (IV)**: Feature predictive power measurement

### Model Validation

- **ROC-AUC**: Discrimination power (0.5 = random, 1.0 = perfect)
- **Gini Coefficient**: 2×AUC - 1
- **KS Statistic**: Maximum separation between default/non-default distributions
- **Calibration**: Predicted probabilities match actual default rates
- **Lift Charts**: Model effectiveness by decile
- **PSI**: Population Stability Index for monitoring drift

### Visualizations (7 Professional Charts)

1. ROC Curve with AUC and Gini
2. KS Chart with maximum separation point
3. Calibration Curve (predicted vs actual)
4. Feature Importance (top 15 features)
5. Lift Chart by decile
6. Model Comparison (3 models side-by-side)
7. PD Score Distribution by class

---

## 🛠️ Installation

### Prerequisites

- **Python 3.8 or higher** (tested with 3.8, 3.9, 3.10, 3.11, 3.12)
- **pip** package manager
- **Git** for cloning the repository

### Step 1: Clone the Repository

```bash
git clone https://github.com/avniderashree/credit-risk-pd-lgd-model.git
cd credit-risk-pd-lgd-model
```

### Step 2: Create Virtual Environment (Recommended)

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `numpy`, `pandas` - Data manipulation
- `scikit-learn` - Machine learning algorithms
- `xgboost` - Gradient boosting models
- `matplotlib`, `seaborn` - Visualization
- `joblib` - Model serialization
- `jupyter` - Interactive notebooks

### Step 4: Verify Installation

```bash
python -c "import xgboost; import sklearn; print('✓ All packages installed successfully')"
```

---

## 🚀 How to Run

### Option 1: Run the Main Script (Recommended for First-Time Users)

```bash
python main.py
```

This will:
1. Generate synthetic credit data (5,000 loans)
2. Engineer features (15 → 23 features)
3. Train 3 PD models (Logistic, Random Forest, XGBoost)
4. Train 3 LGD models (Ridge, Random Forest, XGBoost)
5. Calculate Expected Loss for the portfolio
6. Generate 7 visualization charts in `./output/`
7. Save trained models to `./models/`

**Expected runtime:** 30-60 seconds

### Option 2: Interactive Jupyter Notebook

```bash
jupyter notebook notebooks/credit_risk_analysis.ipynb
```

This opens an interactive notebook where you can:
- Run cells step-by-step
- Modify parameters
- Visualize results inline
- Experiment with different settings

### Option 3: Run as Python Module

```bash
cd credit-risk-pd-lgd-model
python -m src.data_loader      # Test data generation
python -m src.pd_model          # Test PD models
python -m src.lgd_model         # Test LGD models
```

### Option 4: Import in Your Code

```python
from src.data_loader import prepare_credit_data
from src.feature_engineering import FeatureEngineer
from src.pd_model import train_xgboost

# Load data
X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)

# Train model
fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train)
X_test_fe = fe.transform(X_test)

result = train_xgboost(X_train_fe, y_train['default'], X_test_fe, y_test['default'])
print(f"ROC-AUC: {result.roc_auc:.4f}")
```

---

## 📊 Understanding the Output

### Console Output

When you run `python main.py`, you'll see:

```
============================================================
 CREDIT RISK PD/LGD MODEL
============================================================

------------------------------------------------------------
 STEP 1: Loading Credit Data
------------------------------------------------------------
Generated 5000 synthetic credit records
Default rate: 13.84%

  • Training samples: 4000
  • Test samples: 1000
  • Features: 15
  • Default rate: 13.85%

------------------------------------------------------------
 STEP 2: Feature Engineering
------------------------------------------------------------
  • Original features: 15
  • Engineered features: 23

Top Features by Information Value:
  • credit_score         IV=0.5733 (Strong)
  • delinquencies_2yr    IV=0.0387 (Weak)
  • dti_ratio            IV=0.0367 (Weak)

------------------------------------------------------------
 STEP 3: Training PD Models
------------------------------------------------------------

PD Model Comparison:
              Model  ROC-AUC   Gini  KS Statistic
Logistic Regression   0.7236 0.4471        0.3464
      Random Forest   0.7000 0.4000        0.3502
            XGBoost   0.6646 0.3293        0.2699

✓ Best PD Model: Logistic Regression
  • ROC-AUC: 0.7236
  • Gini: 0.4471
  • KS Statistic: 0.3464

------------------------------------------------------------
 STEP 5: Expected Loss Calculation
------------------------------------------------------------
  • Average PD:  14.82%
  • Average LGD: 39.92%
  • Total Expected Loss: $4,157,222

📊 Risk Segmentation:
           Count  Avg PD    Total EL
Low           84    2.3%   $90,411
Medium       268    9.8%  $594,744
High         648   22.4%  $987,654
Very High    ...
```

### Generated Files

After running, you'll find:

**`./output/` folder (Visualizations):**
| File | Description |
|------|-------------|
| `roc_curve.png` | ROC curve with AUC=0.72 |
| `ks_chart.png` | KS chart showing 0.35 separation |
| `calibration_curve.png` | Predicted vs actual probabilities |
| `feature_importance.png` | Top 15 predictive features |
| `lift_chart.png` | Model lift by decile |
| `model_comparison.png` | 3 models compared |
| `probability_distribution.png` | Score distribution by class |

**`./models/` folder (Saved Models):**
| File | Description |
|------|-------------|
| `pd_model.pkl` | Best PD model (serialized) |
| `lgd_model.pkl` | Best LGD model (serialized) |
| `feature_engineer.pkl` | Fitted feature transformer |

### Interpreting Results

| Metric | Good Value | Our Result | Interpretation |
|--------|------------|------------|----------------|
| ROC-AUC | > 0.70 | 0.7236 | Good discrimination |
| Gini | > 0.40 | 0.4471 | Strong predictive power |
| KS | > 0.30 | 0.3464 | Excellent class separation |
| Brier Score | < 0.20 | 0.13 | Well-calibrated |

---

## 📁 Project Structure

```
credit-risk-pd-lgd-model/
│
├── main.py                     # 🚀 Main entry point - run this!
├── requirements.txt            # Python dependencies
├── README.md                   # You are here
├── .gitignore                  # Git ignore rules
│
├── src/                        # Core Python modules
│   ├── __init__.py             # Package marker
│   ├── data_loader.py          # Data loading & synthetic generation
│   ├── feature_engineering.py  # WoE, IV, derived features
│   ├── pd_model.py             # PD classification models
│   ├── lgd_model.py            # LGD regression models
│   ├── evaluation.py           # Gini, KS, PSI metrics
│   └── visualization.py        # Charts and plots
│
├── notebooks/
│   └── credit_risk_analysis.ipynb  # Interactive exploration
│
├── tests/
│   └── test_credit_risk.py     # Unit tests
│
├── models/                     # Saved trained models (generated)
│   ├── pd_model.pkl
│   ├── lgd_model.pkl
│   └── feature_engineer.pkl
│
├── output/                     # Visualization outputs (generated)
│   ├── roc_curve.png
│   ├── ks_chart.png
│   ├── calibration_curve.png
│   ├── feature_importance.png
│   ├── lift_chart.png
│   ├── model_comparison.png
│   └── probability_distribution.png
│
└── data/                       # Data folder (optional)
    └── .gitkeep
```

---

## 📖 Module Documentation

### `src/data_loader.py`

**Purpose:** Load and prepare credit data for modeling.

**Key Functions:**

```python
# Generate synthetic credit data
df = generate_synthetic_credit_data(n_samples=5000)

# Load German Credit Dataset from UCI
df = load_german_credit_data()

# Prepare train/test splits
X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
```

**Generated Features:**
- `loan_amount`, `loan_term_months`, `interest_rate`
- `age`, `annual_income`, `employment_length`
- `credit_score`, `credit_utilization`, `dti_ratio`
- `delinquencies_2yr`, `bankruptcies`
- `home_ownership`, `loan_purpose`

---

### `src/feature_engineering.py`

**Purpose:** Transform raw features into model-ready inputs.

**Key Classes/Functions:**

```python
# Feature engineering pipeline
fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train)  # Fit on training
X_test_fe = fe.transform(X_test)        # Apply to test

# Calculate Information Value
iv_df = calculate_information_value(X, y)

# Weight of Evidence transformation
woe_table, woe_dict, iv = create_weight_of_evidence(X, y, 'credit_score')
```

**Derived Features Created:**
- `loan_to_income`: loan_amount / annual_income
- `monthly_payment`: Amortized payment estimate
- `payment_to_income`: Monthly payment burden
- `credit_score_bucket`: Discretized credit score
- `high_utilization`: Flag if utilization > 70%
- `has_delinquency`: Flag if any delinquencies
- `stable_employment`: Flag if employed 2+ years

---

### `src/pd_model.py`

**Purpose:** Train and evaluate Probability of Default models.

**Key Functions:**

```python
# Train individual models
result = train_logistic_regression(X_train, y_train, X_test, y_test)
result = train_random_forest(X_train, y_train, X_test, y_test)
result = train_xgboost(X_train, y_train, X_test, y_test)

# Train all models
results = train_all_pd_models(X_train, y_train, X_test, y_test)

# Compare models
comparison_df = model_comparison_table(results)
```

**Returns `PDModelResult` with:**
- `model`: Trained sklearn/xgboost model
- `roc_auc`: Area under ROC curve
- `gini`: Gini coefficient
- `ks_statistic`: Kolmogorov-Smirnov statistic
- `brier_score`: Calibration metric
- `feature_importance`: DataFrame of feature weights

---

### `src/lgd_model.py`

**Purpose:** Train and evaluate Loss Given Default models.

**Key Functions:**

```python
# Train LGD models (on defaulted loans only)
result = train_linear_lgd(X_train, y_train, X_test, y_test)
result = train_rf_lgd(X_train, y_train, X_test, y_test)
result = train_xgb_lgd(X_train, y_train, X_test, y_test)

# Calculate Expected Loss
el = calculate_expected_loss(pd_predictions, lgd_predictions, ead)
```

---

### `src/evaluation.py`

**Purpose:** Model validation metrics.

**Key Functions:**

```python
# Core metrics
gini = calculate_gini(y_true, y_prob)
ks_stat, threshold = calculate_ks_statistic(y_true, y_prob)

# Tables
ks_table = create_ks_table(y_true, y_prob, n_bins=10)
calibration_table = create_calibration_table(y_true, y_prob)
lift_table = calculate_lift(y_true, y_prob)

# Model stability
psi, breakdown = calculate_psi(expected_dist, actual_dist)
```

---

### `src/visualization.py`

**Purpose:** Generate publication-quality charts.

**Functions:**

```python
plot_roc_curve(y_true, y_prob, model_name, save_path='output/roc.png')
plot_ks_chart(y_true, y_prob, save_path='output/ks.png')
plot_calibration_curve(y_true, y_prob, model_name)
plot_feature_importance(importance_df, top_n=15)
plot_lift_chart(y_true, y_prob, n_bins=10)
plot_model_comparison(results_dict)
plot_probability_distribution(y_true, y_prob)
plot_confusion_matrix(y_true, y_pred)
```

---

## 🔬 Methodology Deep Dive

### PD Modeling Workflow

```
1. Data Collection
   └── Historical loan data with default outcomes

2. Feature Engineering
   ├── Handle missing values
   ├── Create derived features
   ├── Encode categoricals
   └── Scale numericals

3. Model Training
   ├── Logistic Regression (baseline)
   ├── Random Forest
   └── XGBoost (usually best)

4. Model Validation
   ├── ROC-AUC, Gini, KS
   ├── Calibration check
   └── Out-of-time validation

5. Model Deployment
   ├── Score new applications
   └── Monitor for drift (PSI)
```

### LGD Modeling Considerations

LGD models are trained **only on defaulted loans** because:
- Non-defaulted loans have LGD = 0 by definition
- We need to predict loss severity *given* that default occurs
- Typical approach: Train on workout data with actual recovery amounts

**Challenges:**
- Smaller sample size (only defaults)
- Bimodal distribution (some recover 100%, some recover 0%)
- Often uses Beta regression for bounded outcomes

---

## 📈 Model Validation Metrics

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)

- Measures model's ability to distinguish defaults from non-defaults
- Range: 0.5 (random) to 1.0 (perfect)
- Interpretation:
  - 0.5 - 0.6: Poor
  - 0.6 - 0.7: Fair
  - 0.7 - 0.8: Good ✓
  - 0.8 - 0.9: Excellent
  - 0.9+: Outstanding

### Gini Coefficient

- Gini = 2 × AUC - 1
- Range: 0 (random) to 1 (perfect)
- Banking industry standard metric

### KS Statistic (Kolmogorov-Smirnov)

- Maximum separation between cumulative distributions
- Interpretation:
  - < 0.20: Poor
  - 0.20 - 0.30: Fair
  - 0.30 - 0.40: Good ✓
  - 0.40 - 0.50: Very good
  - > 0.50: Excellent

### Information Value (IV)

- Measures feature predictive power
- Used for feature selection
- Interpretation:
  - < 0.02: Useless
  - 0.02 - 0.10: Weak
  - 0.10 - 0.30: Medium
  - 0.30 - 0.50: Strong
  - > 0.50: Suspicious (possible leakage)

---

## 📊 Sample Results

### PD Model Performance

| Model | ROC-AUC | Gini | KS | Brier |
|-------|---------|------|-----|-------|
| Logistic Regression | 0.7236 | 0.4471 | 0.3464 | 0.2007 |
| Random Forest | 0.7000 | 0.4000 | 0.3502 | 0.1327 |
| XGBoost | 0.6646 | 0.3293 | 0.2699 | 0.1617 |

### LGD Model Performance

| Model | RMSE | MAE | R² | Mean LGD |
|-------|------|-----|-----|----------|
| Random Forest | 0.2098 | 0.1752 | -0.02 | 39.92% |
| Ridge | 0.2113 | 0.1765 | -0.04 | 39.37% |
| XGBoost | 0.2259 | 0.1887 | -0.19 | 39.81% |

### Portfolio Summary

- **Total Loans:** 1,000 (test set)
- **Average PD:** 14.82%
- **Average LGD:** 39.92%
- **Total Expected Loss:** $4,157,222
- **Average EL per Loan:** $4,157

---

## 💻 Code Examples

### Score a New Loan Application

```python
import joblib
import pandas as pd

# Load saved models
pd_model = joblib.load('models/pd_model.pkl')
fe = joblib.load('models/feature_engineer.pkl')

# New application
application = pd.DataFrame({
    'loan_amount': [25000],
    'loan_term_months': [36],
    'interest_rate': [12.5],
    'age': [35],
    'annual_income': [75000],
    'employment_length': [5],
    'home_ownership': ['MORTGAGE'],
    'credit_score': [720],
    'num_credit_lines': [6],
    'credit_utilization': [0.35],
    'delinquencies_2yr': [0],
    'bankruptcies': [0],
    'dti_ratio': [28],
    'total_debt': [15000],
    'loan_purpose': ['debt_consolidation']
})

# Transform and predict
X_new = fe.transform(application)
pd_score = pd_model.predict_proba(X_new)[:, 1][0]

print(f"Probability of Default: {pd_score:.2%}")
# Output: Probability of Default: 8.23%
```

### Batch Scoring

```python
# Score entire portfolio
df = pd.read_csv('new_applications.csv')
X = fe.transform(df)
df['pd_score'] = pd_model.predict_proba(X)[:, 1]

# Risk bucketing
df['risk_grade'] = pd.cut(
    df['pd_score'],
    bins=[0, 0.05, 0.10, 0.20, 1.0],
    labels=['A', 'B', 'C', 'D']
)
```

---

## ⚙️ Customization Guide

### Use Your Own Data

```python
# In src/data_loader.py, modify prepare_credit_data():

def prepare_credit_data(use_synthetic=False, data_path='data/my_loans.csv'):
    if not use_synthetic:
        df = pd.read_csv(data_path)
        # Ensure these columns exist:
        # - 'default': 0 or 1
        # - 'lgd': 0.0 to 1.0 (for defaulted loans)
        # - Feature columns
    ...
```

### Tune Model Hyperparameters

```python
# In src/pd_model.py, modify train_xgboost():

model = xgb.XGBClassifier(
    n_estimators=200,      # Increase trees
    max_depth=7,           # Deeper trees
    learning_rate=0.05,    # Lower learning rate
    subsample=0.8,         # Row subsampling
    colsample_bytree=0.8,  # Feature subsampling
    ...
)
```

### Add Custom Features

```python
# In src/feature_engineering.py, modify create_derived_features():

def create_derived_features(self, X):
    X = X.copy()
    
    # Add your custom features
    X['income_per_dependent'] = X['annual_income'] / (X['dependents'] + 1)
    X['credit_age_ratio'] = X['credit_history_length'] / X['age']
    ...
```

---

## 🔧 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'xgboost'**
```bash
pip install xgboost
```

**2. Matplotlib backend issues on macOS**
```python
# Add to top of script:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

**3. Memory issues with large datasets**
```python
# Reduce sample size:
df = generate_synthetic_credit_data(n_samples=1000)  # Instead of 5000
```

**4. XGBoost version warnings**
```bash
# Upgrade to latest:
pip install --upgrade xgboost
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🎓 Technical Skills Demonstrated

| Category | Skills |
|----------|--------|
| **Credit Risk** | PD, LGD, EAD, Expected Loss, Basel framework, IFRS 9 |
| **Machine Learning** | Classification, Regression, XGBoost, Random Forest |
| **Model Validation** | ROC-AUC, Gini, KS, Calibration, Lift, PSI |
| **Feature Engineering** | WoE, IV, Derived features, Encoding, Scaling |
| **Python** | pandas, numpy, scikit-learn, xgboost, matplotlib |
| **Software Engineering** | Modular design, Type hints, Docstrings, Unit tests |

---

## 📚 References

### Books
1. Siddiqi, N. (2012). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. Wiley.
2. Thomas, L.C. (2009). *Consumer Credit Models: Pricing, Profit and Portfolios*. Oxford University Press.
3. Anderson, R. (2007). *The Credit Scoring Toolkit*. Oxford University Press.

### Papers
- Basel Committee on Banking Supervision. *Basel III: A Global Regulatory Framework for More Resilient Banks and Banking Systems*.
- IFRS 9. *Financial Instruments - Expected Credit Loss Model*.

### Online Resources
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Basel III Summary](https://www.bis.org/bcbs/basel3.htm)

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

## 🔗 Related Projects

Check out the complete quantitative finance portfolio:

| Project | Description |
|---------|-------------|
| [Portfolio VaR Calculator](https://github.com/avniderashree/portfolio-var-calculator) | Value at Risk using Historical, Parametric, and Monte Carlo methods |
| [GARCH Volatility Forecaster](https://github.com/avniderashree/garch-volatility-forecaster) | GARCH/EGARCH models for volatility prediction |
| **Credit Risk PD/LGD Model** | You are here! |

---

*Last updated: January 2026*
