"""
Model Evaluation Module
Comprehensive evaluation metrics for credit risk models.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve


def calculate_gini(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Gini coefficient.
    
    Gini = 2 * AUC - 1
    
    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    
    Returns:
    --------
    float
        Gini coefficient
    """
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Calculate Kolmogorov-Smirnov statistic and optimal threshold.
    
    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    
    Returns:
    --------
    Tuple[float, float]
        KS statistic and threshold at KS
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ks_values = tpr - fpr
    ks_stat = ks_values.max()
    ks_threshold = thresholds[ks_values.argmax()]
    
    return ks_stat, ks_threshold


def create_ks_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Create KS table with decile analysis.
    
    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    n_bins : int
        Number of deciles
    
    Returns:
    --------
    pd.DataFrame
        KS table with cumulative metrics
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['decile'] = pd.qcut(df['y_prob'], q=n_bins, labels=False, duplicates='drop')
    df = df.sort_values('decile', ascending=False)
    
    # Aggregate by decile
    agg = df.groupby('decile').agg({
        'y_true': ['sum', 'count'],
        'y_prob': ['min', 'max', 'mean']
    }).reset_index()
    
    agg.columns = ['decile', 'events', 'total', 'min_prob', 'max_prob', 'avg_prob']
    agg = agg.sort_values('decile', ascending=False).reset_index(drop=True)
    
    agg['non_events'] = agg['total'] - agg['events']
    agg['event_rate'] = agg['events'] / agg['total']
    
    total_events = agg['events'].sum()
    total_non_events = agg['non_events'].sum()
    
    agg['cum_events'] = agg['events'].cumsum()
    agg['cum_non_events'] = agg['non_events'].cumsum()
    
    agg['cum_event_pct'] = agg['cum_events'] / total_events
    agg['cum_non_event_pct'] = agg['cum_non_events'] / total_non_events
    
    agg['ks'] = agg['cum_event_pct'] - agg['cum_non_event_pct']
    
    return agg


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> Tuple[float, pd.DataFrame]:
    """
    Calculate Population Stability Index (PSI).
    
    PSI measures distribution shift between training and scoring.
    
    Parameters:
    -----------
    expected : array
        Training/baseline distribution
    actual : array
        Scoring/current distribution
    n_bins : int
        Number of bins
    
    Returns:
    --------
    Tuple[float, pd.DataFrame]
        PSI value and detailed breakdown
    """
    # Create bins from expected distribution
    _, bin_edges = pd.qcut(expected, q=n_bins, retbins=True, duplicates='drop')
    
    # Bin both distributions
    expected_bins = pd.cut(expected, bins=bin_edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=bin_edges, include_lowest=True)
    
    # Calculate percentages
    expected_pct = expected_bins.value_counts(normalize=True).sort_index()
    actual_pct = actual_bins.value_counts(normalize=True).sort_index()
    
    # Align indices
    all_bins = expected_pct.index.union(actual_pct.index)
    expected_pct = expected_pct.reindex(all_bins, fill_value=0.0001)
    actual_pct = actual_pct.reindex(all_bins, fill_value=0.0001)
    
    # Replace zeros
    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct = actual_pct.replace(0, 0.0001)
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = psi_values.sum()
    
    # Create breakdown table
    breakdown = pd.DataFrame({
        'bin': all_bins.astype(str),
        'expected_pct': expected_pct.values,
        'actual_pct': actual_pct.values,
        'psi': psi_values.values
    })
    
    return psi_total, breakdown


def psi_interpretation(psi: float) -> str:
    """
    Interpret PSI value.
    
    Parameters:
    -----------
    psi : float
        PSI value
    
    Returns:
    --------
    str
        Interpretation
    """
    if psi < 0.1:
        return "No significant change"
    elif psi < 0.25:
        return "Moderate change - monitoring required"
    else:
        return "Significant change - model review required"


def create_calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Create calibration table comparing predicted vs actual rates.
    
    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    n_bins : int
        Number of bins
    
    Returns:
    --------
    pd.DataFrame
        Calibration table
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['bin'] = pd.qcut(df['y_prob'], q=n_bins, labels=False, duplicates='drop')
    
    agg = df.groupby('bin').agg({
        'y_true': 'mean',
        'y_prob': 'mean'
    }).reset_index()
    
    agg.columns = ['bin', 'actual_rate', 'predicted_rate']
    agg['calibration_error'] = agg['actual_rate'] - agg['predicted_rate']
    
    return agg


def calculate_lift(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Calculate lift by decile.
    
    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    n_bins : int
        Number of deciles
    
    Returns:
    --------
    pd.DataFrame
        Lift table
    """
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['decile'] = pd.qcut(df['y_prob'], q=n_bins, labels=False, duplicates='drop')
    df = df.sort_values('decile', ascending=False)
    
    base_rate = df['y_true'].mean()
    
    agg = df.groupby('decile').agg({
        'y_true': ['mean', 'sum', 'count'],
        'y_prob': 'mean'
    }).reset_index()
    
    agg.columns = ['decile', 'event_rate', 'events', 'total', 'avg_prob']
    agg = agg.sort_values('decile', ascending=False).reset_index(drop=True)
    
    agg['lift'] = agg['event_rate'] / base_rate
    agg['cum_events'] = agg['events'].cumsum()
    agg['cum_total'] = agg['total'].cumsum()
    agg['cum_event_rate'] = agg['cum_events'] / agg['cum_total']
    agg['cum_lift'] = agg['cum_event_rate'] / base_rate
    
    return agg


def full_model_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model"
) -> Dict:
    """
    Comprehensive model evaluation.
    
    Parameters:
    -----------
    y_true : array
        True binary labels
    y_prob : array
        Predicted probabilities
    model_name : str
        Name for reporting
    
    Returns:
    --------
    Dict
        Dictionary of all evaluation metrics
    """
    # Core metrics
    roc_auc = roc_auc_score(y_true, y_prob)
    gini = calculate_gini(y_true, y_prob)
    ks_stat, ks_threshold = calculate_ks_statistic(y_true, y_prob)
    
    # Tables
    ks_table = create_ks_table(y_true, y_prob)
    calibration_table = create_calibration_table(y_true, y_prob)
    lift_table = calculate_lift(y_true, y_prob)
    
    return {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'gini': gini,
        'ks_statistic': ks_stat,
        'ks_threshold': ks_threshold,
        'ks_table': ks_table,
        'calibration_table': calibration_table,
        'lift_table': lift_table
    }


if __name__ == "__main__":
    print("Testing Model Evaluation...")
    
    # Generate sample data
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.15, n)
    y_prob = np.clip(y_true * 0.7 + np.random.uniform(0, 0.3, n), 0, 1)
    
    # Calculate metrics
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Gini: {calculate_gini(y_true, y_prob):.4f}")
    
    ks, ks_thresh = calculate_ks_statistic(y_true, y_prob)
    print(f"KS Statistic: {ks:.4f} at threshold {ks_thresh:.4f}")
    
    print("\nKS Table:")
    print(create_ks_table(y_true, y_prob).head())
