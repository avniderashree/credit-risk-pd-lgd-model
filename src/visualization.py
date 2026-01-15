"""
Visualization Module
Charts and plots for credit risk modeling.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve


# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve with AUC and Gini.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = np.trapz(tpr, fpr)
    gini = 2 * auc - 1
    
    ax.plot(fpr, tpr, color='navy', linewidth=2, 
            label=f'{model_name} (AUC={auc:.3f}, Gini={gini:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random (AUC=0.5)')
    
    ax.fill_between(fpr, tpr, alpha=0.2, color='navy')
    
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ks_chart(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot KS chart showing separation between classes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort by probability
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob').reset_index(drop=True)
    df['pct'] = (df.index + 1) / len(df)
    
    # Calculate cumulative distributions
    total_events = df['y_true'].sum()
    total_non_events = len(df) - total_events
    
    df['cum_events'] = df['y_true'].cumsum() / total_events
    df['cum_non_events'] = (1 - df['y_true']).cumsum() / total_non_events
    
    # Find max KS
    df['ks'] = abs(df['cum_events'] - df['cum_non_events'])
    max_ks_idx = df['ks'].idxmax()
    max_ks = df.loc[max_ks_idx, 'ks']
    max_ks_pct = df.loc[max_ks_idx, 'pct']
    
    # Plot
    ax.plot(df['pct'], df['cum_events'], color='red', linewidth=2, label='Defaults')
    ax.plot(df['pct'], df['cum_non_events'], color='blue', linewidth=2, label='Non-Defaults')
    
    # KS line
    ax.axvline(x=max_ks_pct, color='green', linestyle='--', linewidth=1.5)
    ax.annotate(f'KS = {max_ks:.3f}', xy=(max_ks_pct, 0.5), 
                fontsize=12, color='green', fontweight='bold')
    
    ax.set_xlabel('Population Percentile', fontsize=11)
    ax.set_ylabel('Cumulative %', fontsize=11)
    ax.set_title('KS Chart', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    n_bins: int = 10,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot calibration curve.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    ax.plot(prob_pred, prob_true, 's-', color='navy', linewidth=2, 
            markersize=8, label=model_name)
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfectly Calibrated')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax.set_ylabel('Fraction of Positives', fontsize=11)
    ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance bar chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n).sort_values('importance')
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_features)))
    
    ax.barh(top_features['feature'], top_features['importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_lift_chart(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot lift chart by decile.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df['decile'] = pd.qcut(df['y_prob'], q=n_bins, labels=False, duplicates='drop')
    
    base_rate = df['y_true'].mean()
    
    agg = df.groupby('decile')['y_true'].mean().sort_index(ascending=False)
    lift = agg / base_rate
    
    deciles = range(1, len(lift) + 1)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(lift)))
    
    ax.bar(deciles, lift.values, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=1, color='navy', linestyle='--', linewidth=2, label='Baseline (Lift=1)')
    
    ax.set_xlabel('Decile (Highest Risk → Lowest)', fontsize=11)
    ax.set_ylabel('Lift', fontsize=11)
    ax.set_title('Lift Chart by Decile', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (d, l) in enumerate(zip(deciles, lift.values)):
        ax.annotate(f'{l:.2f}', xy=(d, l), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results: Dict,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot model comparison across metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    models = list(results.keys())
    metrics = ['roc_auc', 'gini', 'ks_statistic']
    titles = ['ROC-AUC', 'Gini Coefficient', 'KS Statistic']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        values = [getattr(results[m], metric) for m in models]
        model_names = [results[m].model_name for m in models]
        
        bars = ax.bar(model_names, values, color=color, edgecolor='black', linewidth=0.5)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, val),
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('PD Model Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot probability distribution by class.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate by class
    prob_default = y_prob[y_true == 1]
    prob_non_default = y_prob[y_true == 0]
    
    ax.hist(prob_non_default, bins=50, alpha=0.7, color='blue', 
            label=f'Non-Default (n={len(prob_non_default)})', density=True)
    ax.hist(prob_default, bins=50, alpha=0.7, color='red', 
            label=f'Default (n={len(prob_default)})', density=True)
    
    ax.set_xlabel('Predicted Probability', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('PD Score Distribution by Actual Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    print("Visualization module loaded successfully.")
    print("Available functions:")
    print("  - plot_roc_curve()")
    print("  - plot_ks_chart()")
    print("  - plot_calibration_curve()")
    print("  - plot_feature_importance()")
    print("  - plot_lift_chart()")
    print("  - plot_confusion_matrix()")
    print("  - plot_model_comparison()")
    print("  - plot_probability_distribution()")
