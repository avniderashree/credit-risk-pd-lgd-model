"""
Unit Tests for Credit Risk PD/LGD Model
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import generate_synthetic_credit_data, prepare_credit_data
from src.feature_engineering import FeatureEngineer
from src.pd_model import (
    train_logistic_regression, train_xgboost,
    calculate_ks_statistic, PDModelResult
)
from src.lgd_model import train_linear_lgd, LGDModelResult
from src.evaluation import calculate_gini, create_ks_table, calculate_psi


class TestDataLoader:
    """Test data loading functions."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic credit data generation."""
        df = generate_synthetic_credit_data(100)
        
        assert len(df) == 100
        assert 'default' in df.columns
        assert 'lgd' in df.columns
        assert 'loan_amount' in df.columns
    
    def test_default_rate_reasonable(self):
        """Test that default rate is reasonable."""
        df = generate_synthetic_credit_data(1000)
        default_rate = df['default'].mean()
        
        assert 0.05 < default_rate < 0.40  # Reasonable range
    
    def test_prepare_credit_data_split(self):
        """Test data split."""
        X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
        
        assert len(X_train) > len(X_test)
        assert len(X_train) == len(y_train)
        assert 'default' in y_train.columns


class TestFeatureEngineering:
    """Test feature engineering."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
        return X_train, X_test, y_train, y_test
    
    def test_feature_engineer_fit_transform(self, sample_data):
        """Test feature engineering pipeline."""
        X_train, _, _, _ = sample_data
        
        fe = FeatureEngineer()
        X_transformed = fe.fit_transform(X_train)
        
        assert X_transformed.shape[0] == X_train.shape[0]
        assert X_transformed.shape[1] >= X_train.shape[1]  # Should have at least same features
    
    def test_feature_engineer_transform(self, sample_data):
        """Test transform on new data."""
        X_train, X_test, _, _ = sample_data
        
        fe = FeatureEngineer()
        fe.fit(X_train)
        X_test_transformed = fe.transform(X_test)
        
        assert X_test_transformed.shape[0] == X_test.shape[0]


class TestPDModel:
    """Test PD model functions."""
    
    @pytest.fixture
    def prepared_data(self):
        """Prepare data for PD modeling."""
        X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
        
        fe = FeatureEngineer()
        X_train_fe = fe.fit_transform(X_train)
        X_test_fe = fe.transform(X_test)
        
        return X_train_fe, X_test_fe, y_train['default'], y_test['default']
    
    def test_logistic_regression_trains(self, prepared_data):
        """Test logistic regression training."""
        X_train, X_test, y_train, y_test = prepared_data
        
        result = train_logistic_regression(X_train, y_train, X_test, y_test)
        
        assert isinstance(result, PDModelResult)
        assert 0 < result.roc_auc <= 1
        assert result.model is not None
    
    def test_ks_statistic_calculation(self):
        """Test KS statistic calculation."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.2, 1000)
        y_prob = np.clip(y_true * 0.7 + np.random.uniform(0, 0.3, 1000), 0, 1)
        
        ks = calculate_ks_statistic(y_true, y_prob)
        
        assert 0 < ks < 1
    
    def test_gini_coefficient(self):
        """Test Gini coefficient."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.2, 1000)
        y_prob = np.clip(y_true * 0.8 + np.random.uniform(0, 0.2, 1000), 0, 1)
        
        gini = calculate_gini(y_true, y_prob)
        
        assert -1 <= gini <= 1


class TestLGDModel:
    """Test LGD model functions."""
    
    @pytest.fixture
    def lgd_data(self):
        """Prepare data for LGD modeling."""
        df = generate_synthetic_credit_data(500)
        
        # Filter to defaulted
        defaulted = df[df['default'] == 1]
        
        from sklearn.model_selection import train_test_split
        X = defaulted.drop(['default', 'lgd', 'ead', 'expected_loss'], axis=1)
        y = defaulted['lgd']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        fe = FeatureEngineer()
        X_train_fe = fe.fit_transform(X_train)
        X_test_fe = fe.transform(X_test)
        
        return X_train_fe, X_test_fe, y_train, y_test
    
    def test_linear_lgd_trains(self, lgd_data):
        """Test linear LGD training."""
        X_train, X_test, y_train, y_test = lgd_data
        
        result = train_linear_lgd(X_train, y_train, X_test, y_test)
        
        assert isinstance(result, LGDModelResult)
        assert result.rmse >= 0
        assert 0 <= result.mean_lgd_pred <= 1


class TestEvaluation:
    """Test evaluation metrics."""
    
    def test_ks_table_creation(self):
        """Test KS table creation."""
        np.random.seed(42)
        y_true = np.random.binomial(1, 0.2, 1000)
        y_prob = np.clip(y_true * 0.6 + np.random.uniform(0, 0.4, 1000), 0, 1)
        
        ks_table = create_ks_table(y_true, y_prob)
        
        assert 'ks' in ks_table.columns
        assert len(ks_table) <= 10  # Up to 10 deciles
    
    def test_psi_calculation(self):
        """Test PSI calculation."""
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0.1, 1, 1000)  # Slight shift
        
        psi, breakdown = calculate_psi(expected, actual)
        
        assert psi >= 0
        assert 'psi' in breakdown.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
