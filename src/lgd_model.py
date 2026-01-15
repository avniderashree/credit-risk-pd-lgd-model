"""
LGD Model Module
Loss Given Default (LGD) modeling using regression techniques.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@dataclass
class LGDModelResult:
    """Container for LGD model results."""
    model_name: str
    model: object
    rmse: float
    mae: float
    r2: float
    mean_lgd_pred: float
    mean_lgd_actual: float
    feature_importance: Optional[pd.DataFrame]
    
    def __str__(self):
        return (
            f"{self.model_name} Results:\n"
            f"  RMSE:          {self.rmse:.4f}\n"
            f"  MAE:           {self.mae:.4f}\n"
            f"  R²:            {self.r2:.4f}\n"
            f"  Mean LGD Pred: {self.mean_lgd_pred:.2%}\n"
            f"  Mean LGD True: {self.mean_lgd_actual:.2%}"
        )


def train_linear_lgd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> LGDModelResult:
    """
    Train Linear Regression LGD model.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Feature matrices
    y_train, y_test : pd.Series
        LGD targets
    
    Returns:
    --------
    LGDModelResult
        Model results
    """
    model = Ridge(alpha=1.0, **kwargs)
    model.fit(X_train, y_train)
    
    # Predictions (clip to [0, 1])
    y_pred = np.clip(model.predict(X_test), 0, 1)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': np.abs(model.coef_)
    }).sort_values('importance', ascending=False)
    
    return LGDModelResult(
        model_name='Linear Regression (Ridge)',
        model=model,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mean_lgd_pred=y_pred.mean(),
        mean_lgd_actual=y_test.mean(),
        feature_importance=importance_df
    )


def train_rf_lgd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> LGDModelResult:
    """
    Train Random Forest LGD model.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    y_pred = np.clip(model.predict(X_test), 0, 1)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return LGDModelResult(
        model_name='Random Forest',
        model=model,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mean_lgd_pred=y_pred.mean(),
        mean_lgd_actual=y_test.mean(),
        feature_importance=importance_df
    )


def train_xgb_lgd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> LGDModelResult:
    """
    Train XGBoost LGD model.
    """
    if not HAS_XGBOOST:
        print("XGBoost not installed, using Gradient Boosting instead")
        return train_gb_lgd(X_train, y_train, X_test, y_test, **kwargs)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        **kwargs
    )
    
    model.fit(X_train, y_train, verbose=False)
    
    y_pred = np.clip(model.predict(X_test), 0, 1)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return LGDModelResult(
        model_name='XGBoost',
        model=model,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mean_lgd_pred=y_pred.mean(),
        mean_lgd_actual=y_test.mean(),
        feature_importance=importance_df
    )


def train_gb_lgd(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    **kwargs
) -> LGDModelResult:
    """
    Train Gradient Boosting LGD model.
    """
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        **kwargs
    )
    
    model.fit(X_train, y_train)
    
    y_pred = np.clip(model.predict(X_test), 0, 1)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return LGDModelResult(
        model_name='Gradient Boosting',
        model=model,
        rmse=rmse,
        mae=mae,
        r2=r2,
        mean_lgd_pred=y_pred.mean(),
        mean_lgd_actual=y_test.mean(),
        feature_importance=importance_df
    )


def train_all_lgd_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, LGDModelResult]:
    """
    Train all LGD models and compare.
    """
    results = {}
    
    print("Training Linear LGD model...")
    results['linear'] = train_linear_lgd(X_train, y_train, X_test, y_test)
    
    print("Training Random Forest LGD model...")
    results['random_forest'] = train_rf_lgd(X_train, y_train, X_test, y_test)
    
    print("Training XGBoost LGD model...")
    results['xgboost'] = train_xgb_lgd(X_train, y_train, X_test, y_test)
    
    return results


def lgd_comparison_table(results: Dict[str, LGDModelResult]) -> pd.DataFrame:
    """
    Create comparison table of LGD models.
    """
    data = []
    for name, result in results.items():
        data.append({
            'Model': result.model_name,
            'RMSE': f"{result.rmse:.4f}",
            'MAE': f"{result.mae:.4f}",
            'R²': f"{result.r2:.4f}",
            'Mean LGD': f"{result.mean_lgd_pred:.2%}"
        })
    
    return pd.DataFrame(data).sort_values('RMSE')


def calculate_expected_loss(
    pd_predictions: np.ndarray,
    lgd_predictions: np.ndarray,
    ead: np.ndarray
) -> np.ndarray:
    """
    Calculate Expected Loss (EL = PD × LGD × EAD).
    
    Parameters:
    -----------
    pd_predictions : array
        Probability of default predictions
    lgd_predictions : array
        Loss given default predictions
    ead : array
        Exposure at default
    
    Returns:
    --------
    np.ndarray
        Expected loss for each observation
    """
    return pd_predictions * lgd_predictions * ead


if __name__ == "__main__":
    print("Testing LGD Models...")
    
    from data_loader import prepare_credit_data
    from feature_engineering import FeatureEngineer
    
    X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
    
    # Filter to only defaulted loans for LGD modeling
    default_train = y_train['default'] == 1
    default_test = y_test['default'] == 1
    
    X_train_lgd = X_train[default_train]
    y_train_lgd = y_train.loc[default_train, 'lgd']
    X_test_lgd = X_test[default_test]
    y_test_lgd = y_test.loc[default_test, 'lgd']
    
    print(f"LGD training samples: {len(X_train_lgd)}")
    print(f"LGD test samples: {len(X_test_lgd)}")
    
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train_lgd)
    X_test_fe = fe.transform(X_test_lgd)
    
    results = train_all_lgd_models(X_train_fe, y_train_lgd, X_test_fe, y_test_lgd)
    
    print("\n" + "=" * 60)
    print("LGD Model Comparison:")
    print(lgd_comparison_table(results).to_string(index=False))
