"""
Data Loader Module
Loads and preprocesses credit risk dataset.
Uses German Credit Dataset from UCI Machine Learning Repository.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


def load_german_credit_data() -> pd.DataFrame:
    """
    Load the German Credit Dataset.
    
    This dataset contains 1000 loan applications with 20 attributes.
    Target: Good (1) or Bad (2) credit risk.
    
    Returns:
    --------
    pd.DataFrame
        German Credit dataset
    """
    # UCI ML Repository column names
    columns = [
        'checking_account', 'duration_months', 'credit_history', 
        'purpose', 'credit_amount', 'savings_account', 'employment_years',
        'installment_rate', 'personal_status', 'other_debtors',
        'residence_years', 'property', 'age', 'other_installments',
        'housing', 'existing_credits', 'job', 'dependents',
        'telephone', 'foreign_worker', 'target'
    ]
    
    # URL for German Credit Data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    try:
        df = pd.read_csv(url, sep=' ', header=None, names=columns)
        print(f"Data loaded from UCI repository: {len(df)} records")
    except Exception as e:
        print(f"Could not load from URL: {e}")
        print("Generating synthetic credit data instead...")
        df = generate_synthetic_credit_data(1000)
    
    return df


def generate_synthetic_credit_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic credit risk data for demonstration.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    pd.DataFrame
        Synthetic credit dataset
    """
    np.random.seed(42)
    
    # Generate features
    data = {
        # Loan characteristics
        'loan_amount': np.random.lognormal(mean=10, sigma=0.5, size=n_samples).astype(int),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], size=n_samples),
        'interest_rate': np.random.uniform(5, 25, size=n_samples),
        
        # Borrower characteristics
        'age': np.random.randint(21, 70, size=n_samples),
        'annual_income': np.random.lognormal(mean=11, sigma=0.5, size=n_samples).astype(int),
        'employment_length': np.random.exponential(5, size=n_samples).astype(int),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], size=n_samples, p=[0.4, 0.2, 0.4]),
        
        # Credit history
        'credit_score': np.random.normal(680, 80, size=n_samples).astype(int),
        'num_credit_lines': np.random.poisson(5, size=n_samples),
        'credit_utilization': np.random.beta(2, 5, size=n_samples),
        'delinquencies_2yr': np.random.poisson(0.3, size=n_samples),
        'bankruptcies': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        
        # Debt info
        'dti_ratio': np.random.beta(2, 5, size=n_samples) * 50,  # Debt-to-income
        'total_debt': np.random.lognormal(mean=10, sigma=0.8, size=n_samples).astype(int),
        
        # Loan purpose
        'loan_purpose': np.random.choice(
            ['debt_consolidation', 'credit_card', 'home_improvement', 
             'major_purchase', 'medical', 'car', 'other'],
            size=n_samples,
            p=[0.35, 0.2, 0.15, 0.1, 0.05, 0.1, 0.05]
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges
    df['credit_score'] = df['credit_score'].clip(300, 850)
    df['employment_length'] = df['employment_length'].clip(0, 40)
    df['credit_utilization'] = df['credit_utilization'].clip(0, 1)
    
    # Generate target (default) based on features
    # Higher probability of default if: low credit score, high DTI, delinquencies
    default_prob = (
        0.02 +  # Base rate
        0.15 * (df['credit_score'] < 600).astype(float) +
        0.10 * (df['credit_score'] < 650).astype(float) +
        0.08 * (df['dti_ratio'] > 35).astype(float) +
        0.05 * (df['dti_ratio'] > 25).astype(float) +
        0.12 * (df['delinquencies_2yr'] > 0).astype(float) +
        0.15 * (df['delinquencies_2yr'] > 2).astype(float) +
        0.10 * df['bankruptcies'] +
        0.05 * (df['employment_length'] < 2).astype(float) +
        0.03 * (df['credit_utilization'] > 0.7).astype(float)
    )
    
    df['default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Generate LGD (Loss Given Default) for defaulted loans
    # LGD typically follows a beta distribution
    df['lgd'] = np.where(
        df['default'] == 1,
        np.random.beta(2, 3, size=n_samples),  # Mean around 0.4
        0.0
    )
    
    # Calculate EAD (Exposure at Default)
    df['ead'] = df['loan_amount']
    
    # Calculate Expected Loss
    df['expected_loss'] = df['default'] * df['lgd'] * df['ead']
    
    print(f"Generated {n_samples} synthetic credit records")
    print(f"Default rate: {df['default'].mean():.2%}")
    
    return df


def preprocess_german_credit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess German Credit dataset for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw German Credit data
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    df = df.copy()
    
    # Convert target: 1 = Good, 2 = Bad -> 0 = Good, 1 = Bad (default)
    df['default'] = (df['target'] == 2).astype(int)
    df = df.drop('target', axis=1)
    
    # Generate synthetic LGD for defaulted loans
    np.random.seed(42)
    df['lgd'] = np.where(
        df['default'] == 1,
        np.random.beta(2, 3, size=len(df)),
        0.0
    )
    
    # EAD is the credit amount
    df['ead'] = df['credit_amount']
    
    return df


def prepare_credit_data(
    use_synthetic: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare credit data for modeling.
    
    Parameters:
    -----------
    use_synthetic : bool
        Whether to use synthetic data (True) or German Credit (False)
    test_size : float
        Proportion for test set
    random_state : int
        Random seed
    
    Returns:
    --------
    Tuple of (X_train, X_test, y_train, y_test)
    """
    if use_synthetic:
        df = generate_synthetic_credit_data(5000)
    else:
        df = load_german_credit_data()
        df = preprocess_german_credit(df)
    
    # Separate features and target
    target_cols = ['default', 'lgd', 'ead', 'expected_loss'] if 'expected_loss' in df.columns else ['default', 'lgd', 'ead']
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    X = df[feature_cols]
    y = df[['default', 'lgd']]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y['default']
    )
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples ({y_train['default'].mean():.2%} default rate)")
    print(f"  Test: {len(X_test)} samples ({y_test['default'].mean():.2%} default rate)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the module
    print("Testing Data Loader...")
    X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
    
    print(f"\nFeatures: {list(X_train.columns)}")
    print(f"\nTarget distribution:")
    print(y_train['default'].value_counts(normalize=True))
