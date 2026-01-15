"""
Feature Engineering Module
Creates features for credit risk modeling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Feature engineering for credit risk models.
    """
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.label_encoders = {}
    
    def identify_column_types(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify categorical and numerical columns.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        
        Returns:
        --------
        Tuple of (categorical_cols, numerical_cols)
        """
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        return categorical_cols, numerical_cols
    
    def create_derived_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Raw features
        
        Returns:
        --------
        pd.DataFrame
            Features with derived columns
        """
        X = X.copy()
        
        # Loan-to-income ratio
        if 'loan_amount' in X.columns and 'annual_income' in X.columns:
            X['loan_to_income'] = X['loan_amount'] / (X['annual_income'] + 1)
        
        # Monthly payment estimate
        if 'loan_amount' in X.columns and 'loan_term_months' in X.columns:
            if 'interest_rate' in X.columns:
                r = X['interest_rate'] / 100 / 12  # Monthly rate
                n = X['loan_term_months']
                X['monthly_payment'] = X['loan_amount'] * (r * (1 + r)**n) / ((1 + r)**n - 1)
                X['monthly_payment'] = X['monthly_payment'].fillna(X['loan_amount'] / X['loan_term_months'])
            else:
                X['monthly_payment'] = X['loan_amount'] / X['loan_term_months']
        
        # Payment to income ratio
        if 'monthly_payment' in X.columns and 'annual_income' in X.columns:
            X['payment_to_income'] = X['monthly_payment'] * 12 / (X['annual_income'] + 1)
        
        # Credit score buckets
        if 'credit_score' in X.columns:
            X['credit_score_bucket'] = pd.cut(
                X['credit_score'],
                bins=[0, 580, 670, 740, 800, 850],
                labels=['very_poor', 'fair', 'good', 'very_good', 'exceptional']
            ).astype(str)
        
        # Age groups
        if 'age' in X.columns:
            X['age_group'] = pd.cut(
                X['age'],
                bins=[0, 25, 35, 50, 65, 100],
                labels=['young', 'young_adult', 'middle_aged', 'senior', 'elderly']
            ).astype(str)
        
        # High utilization flag
        if 'credit_utilization' in X.columns:
            X['high_utilization'] = (X['credit_utilization'] > 0.7).astype(int)
        
        # Delinquency flag
        if 'delinquencies_2yr' in X.columns:
            X['has_delinquency'] = (X['delinquencies_2yr'] > 0).astype(int)
        
        # Employment stability
        if 'employment_length' in X.columns:
            X['stable_employment'] = (X['employment_length'] >= 2).astype(int)
        
        return X
    
    def fit(self, X: pd.DataFrame) -> 'FeatureEngineer':
        """
        Fit the feature engineering pipeline.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training features
        
        Returns:
        --------
        self
        """
        # Create derived features first
        X_derived = self.create_derived_features(X)
        
        # Identify column types
        self.categorical_cols, self.numerical_cols = self.identify_column_types(X_derived)
        
        # Encode categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            le.fit(X_derived[col].astype(str))
            self.label_encoders[col] = le
        
        # Create scaler for numerical features
        self.scaler = StandardScaler()
        if self.numerical_cols:
            self.scaler.fit(X_derived[self.numerical_cols])
        
        self.feature_names = self.categorical_cols + self.numerical_cols
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted pipeline.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
        
        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        X_derived = self.create_derived_features(X)
        X_transformed = X_derived.copy()
        
        # Encode categoricals
        for col in self.categorical_cols:
            if col in X_transformed.columns:
                # Handle unseen categories
                X_transformed[col] = X_transformed[col].astype(str)
                known_classes = set(self.label_encoders[col].classes_)
                X_transformed[col] = X_transformed[col].apply(
                    lambda x: x if x in known_classes else self.label_encoders[col].classes_[0]
                )
                X_transformed[col] = self.label_encoders[col].transform(X_transformed[col])
        
        # Scale numerical features
        if self.numerical_cols:
            X_transformed[self.numerical_cols] = self.scaler.transform(X_transformed[self.numerical_cols])
        
        # Select only the features we trained on
        all_features = self.categorical_cols + self.numerical_cols
        X_transformed = X_transformed[[c for c in all_features if c in X_transformed.columns]]
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        
        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        return self.fit(X).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation."""
        return self.feature_names


def create_weight_of_evidence(
    X: pd.DataFrame,
    y: pd.Series,
    feature: str,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, dict, float]:
    """
    Calculate Weight of Evidence (WoE) for a feature.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Binary target
    feature : str
        Feature to calculate WoE for
    n_bins : int
        Number of bins for continuous features
    
    Returns:
    --------
    Tuple of (WoE DataFrame, WoE mapping dict)
    """
    df = pd.DataFrame({feature: X[feature], 'target': y})
    
    # Bin continuous features
    if df[feature].dtype in ['int64', 'float64']:
        df['bin'] = pd.qcut(df[feature], q=n_bins, duplicates='drop')
    else:
        df['bin'] = df[feature]
    
    # Calculate WoE
    grouped = df.groupby('bin')['target'].agg(['sum', 'count'])
    grouped.columns = ['events', 'total']
    grouped['non_events'] = grouped['total'] - grouped['events']
    
    total_events = grouped['events'].sum()
    total_non_events = grouped['non_events'].sum()
    
    grouped['dist_events'] = grouped['events'] / total_events
    grouped['dist_non_events'] = grouped['non_events'] / total_non_events
    
    # Avoid log(0)
    grouped['dist_events'] = grouped['dist_events'].replace(0, 0.0001)
    grouped['dist_non_events'] = grouped['dist_non_events'].replace(0, 0.0001)
    
    grouped['woe'] = np.log(grouped['dist_non_events'] / grouped['dist_events'])
    grouped['iv'] = (grouped['dist_non_events'] - grouped['dist_events']) * grouped['woe']
    
    total_iv = grouped['iv'].sum()
    
    woe_dict = grouped['woe'].to_dict()
    
    return grouped, woe_dict, total_iv


def calculate_information_value(
    X: pd.DataFrame,
    y: pd.Series,
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate Information Value for all features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Binary target
    features : List[str], optional
        Features to analyze (default: all)
    
    Returns:
    --------
    pd.DataFrame
        IV for each feature
    """
    if features is None:
        features = X.columns.tolist()
    
    iv_results = []
    
    for feature in features:
        try:
            _, _, iv = create_weight_of_evidence(X, y, feature)
            iv_results.append({'feature': feature, 'iv': iv})
        except Exception:
            iv_results.append({'feature': feature, 'iv': 0})
    
    iv_df = pd.DataFrame(iv_results).sort_values('iv', ascending=False)
    
    # Add interpretation
    def iv_interpretation(iv):
        if iv < 0.02:
            return 'Not useful'
        elif iv < 0.1:
            return 'Weak'
        elif iv < 0.3:
            return 'Medium'
        elif iv < 0.5:
            return 'Strong'
        else:
            return 'Suspicious'
    
    iv_df['interpretation'] = iv_df['iv'].apply(iv_interpretation)
    
    return iv_df


if __name__ == "__main__":
    print("Testing Feature Engineering...")
    
    from data_loader import prepare_credit_data
    X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
    
    # Test feature engineering
    fe = FeatureEngineer()
    X_train_transformed = fe.fit_transform(X_train)
    X_test_transformed = fe.transform(X_test)
    
    print(f"\nOriginal features: {X_train.shape[1]}")
    print(f"Transformed features: {X_train_transformed.shape[1]}")
    print(f"\nFeature names: {fe.get_feature_names()[:10]}...")
    
    # Calculate IV
    print("\nInformation Value:")
    iv_df = calculate_information_value(X_train, y_train['default'])
    print(iv_df.head(10))
