"""
PD Model Module
Probability of Default (PD) modeling using XGBoost and Logistic Regression.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class PDModelResult:
    """Container for PD model results."""
    model_name: str
    model: object
    roc_auc: float
    gini: float
    ks_statistic: float
    brier_score: float
    feature_importance: Optional[pd.DataFrame]
    
    def __str__(self):
        return (
            f"{self.model_name} Results:\n"
            f"  ROC-AUC:      {self.roc_auc:.4f}\n"
            f"  Gini:         {self.gini:.4f}\n"
            f"  KS Statistic: {self.ks_statistic:.4f}\n"
            f"  Brier Score:  {self.brier_score:.4f}"
        )


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic.
    
    KS measures the maximum separation between cumulative distributions
    of defaults and non-defaults.
    
    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    
    Returns:
    --------
    float
        KS statistic
    """
    # Sort by predicted probability
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False).reset_index(drop=True)
    
    # Calculate cumulative distributions
    total_events = df['y_true'].sum()
    total_non_events = len(df) - total_events
    
    # Handle edge cases
    if total_events == 0 or total_non_events == 0:
        return 0.0
    
    df['cum_events'] = df['y_true'].cumsum() / total_events
    df['cum_non_events'] = (1 - df['y_true']).cumsum() / total_non_events
    
    # KS is max difference
    df['ks'] = abs(df['cum_events'] - df['cum_non_events'])
    ks_stat = df['ks'].max()
    
    return ks_stat


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> PDModelResult:
    """
    Train Logistic Regression PD model.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Feature matrices
    y_train, y_test : pd.Series
        Target variables
    
    Returns:
    --------
    PDModelResult
        Model results
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    gini = 2 * roc_auc - 1
    ks_stat = calculate_ks_statistic(y_test.values, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    # Feature importance (coefficients)
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(model.coef_[0])
    }).sort_values('importance', ascending=False)
    
    return PDModelResult(
        model_name='Logistic Regression',
        model=model,
        roc_auc=roc_auc,
        gini=gini,
        ks_statistic=ks_stat,
        brier_score=brier,
        feature_importance=importance_df
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> PDModelResult:
    """
    Train Random Forest PD model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    gini = 2 * roc_auc - 1
    ks_stat = calculate_ks_statistic(y_test.values, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return PDModelResult(
        model_name='Random Forest',
        model=model,
        roc_auc=roc_auc,
        gini=gini,
        ks_statistic=ks_stat,
        brier_score=brier,
        feature_importance=importance_df
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> PDModelResult:
    """
    Train XGBoost PD model.
    """
    if not HAS_XGBOOST:
        print("XGBoost not installed, using Gradient Boosting instead")
        return train_gradient_boosting(X_train, y_train, X_test, y_test, **kwargs)
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        **kwargs
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    gini = 2 * roc_auc - 1
    ks_stat = calculate_ks_statistic(y_test.values, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return PDModelResult(
        model_name='XGBoost',
        model=model,
        roc_auc=roc_auc,
        gini=gini,
        ks_statistic=ks_stat,
        brier_score=brier,
        feature_importance=importance_df
    )


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> PDModelResult:
    """
    Train Gradient Boosting PD model (fallback for XGBoost).
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    gini = 2 * roc_auc - 1
    ks_stat = calculate_ks_statistic(y_test.values, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return PDModelResult(
        model_name='Gradient Boosting',
        model=model,
        roc_auc=roc_auc,
        gini=gini,
        ks_statistic=ks_stat,
        brier_score=brier,
        feature_importance=importance_df
    )


def train_all_pd_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, PDModelResult]:
    """
    Train all PD models and compare.
    
    Returns:
    --------
    Dict[str, PDModelResult]
        Dictionary of model results
    """
    results = {}
    
    print("Training Logistic Regression...")
    results['logistic'] = train_logistic_regression(X_train, y_train, X_test, y_test)
    
    print("Training Random Forest...")
    results['random_forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    
    print("Training XGBoost...")
    results['xgboost'] = train_xgboost(X_train, y_train, X_test, y_test)
    
    return results


def model_comparison_table(results: Dict[str, PDModelResult]) -> pd.DataFrame:
    """
    Create comparison table of PD models.
    """
    data = []
    for name, result in results.items():
        data.append({
            'Model': result.model_name,
            'ROC-AUC': f"{result.roc_auc:.4f}",
            'Gini': f"{result.gini:.4f}",
            'KS Statistic': f"{result.ks_statistic:.4f}",
            'Brier Score': f"{result.brier_score:.4f}"
        })
    
    return pd.DataFrame(data).sort_values('ROC-AUC', ascending=False)


def save_model(result: PDModelResult, path: str):
    """Save trained model to disk."""
    joblib.dump(result.model, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    """Load trained model from disk."""
    return joblib.load(path)


if __name__ == "__main__":
    print("Testing PD Models...")
    
    from data_loader import prepare_credit_data
    from feature_engineering import FeatureEngineer
    
    X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
    
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)
    
    results = train_all_pd_models(X_train_fe, y_train['default'], X_test_fe, y_test['default'])
    
    print("\n" + "=" * 60)
    print("Model Comparison:")
    print(model_comparison_table(results).to_string(index=False))
