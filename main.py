#!/usr/bin/env python3
"""
Credit Risk PD/LGD Model
========================
Main execution script for credit risk modeling.

Author: Avni Derashree
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.data_loader import prepare_credit_data
from src.feature_engineering import FeatureEngineer, calculate_information_value
from src.pd_model import train_all_pd_models, model_comparison_table
from src.lgd_model import train_all_lgd_models, lgd_comparison_table, calculate_expected_loss
from src.evaluation import full_model_evaluation, calculate_ks_statistic, create_ks_table
from src.visualization import (
    plot_roc_curve, plot_ks_chart, plot_calibration_curve,
    plot_feature_importance, plot_lift_chart, plot_model_comparison,
    plot_probability_distribution
)


def print_header(text: str, char: str = "="):
    """Print formatted section header."""
    print(f"\n{char * 60}")
    print(f" {text}")
    print(f"{char * 60}")


def main():
    """Main execution function."""
    
    print_header("CREDIT RISK PD/LGD MODEL", "=")
    print("\nThis analysis builds credit risk models for:")
    print("  1. Probability of Default (PD) - Classification")
    print("  2. Loss Given Default (LGD) - Regression")
    print("  3. Expected Loss (EL = PD × LGD × EAD)")
    
    # =========================================================================
    # STEP 1: Load and Prepare Data
    # =========================================================================
    print_header("STEP 1: Loading Credit Data", "-")
    
    X_train, X_test, y_train, y_test = prepare_credit_data(use_synthetic=True)
    
    print(f"\nDataset Statistics:")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Test samples: {len(X_test)}")
    print(f"  • Features: {X_train.shape[1]}")
    print(f"  • Default rate (train): {y_train['default'].mean():.2%}")
    print(f"  • Default rate (test): {y_test['default'].mean():.2%}")
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print_header("STEP 2: Feature Engineering", "-")
    
    print("\nApplying feature engineering...")
    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)
    
    print(f"  • Original features: {X_train.shape[1]}")
    print(f"  • Engineered features: {X_train_fe.shape[1]}")
    
    # Calculate Information Value
    print("\nTop Features by Information Value:")
    iv_df = calculate_information_value(X_train, y_train['default'])
    top_iv = iv_df.head(10)
    
    for _, row in top_iv.iterrows():
        print(f"  • {row['feature']:<25} IV={row['iv']:.4f} ({row['interpretation']})")
    
    # =========================================================================
    # STEP 3: Train PD Models
    # =========================================================================
    print_header("STEP 3: Training PD Models", "-")
    
    pd_results = train_all_pd_models(
        X_train_fe, y_train['default'], 
        X_test_fe, y_test['default']
    )
    
    print("\n" + "-" * 60)
    print("PD Model Comparison:")
    print("-" * 60)
    print(model_comparison_table(pd_results).to_string(index=False))
    
    # Select best model
    best_pd_name = max(pd_results, key=lambda x: pd_results[x].roc_auc)
    best_pd = pd_results[best_pd_name]
    
    print(f"\n✓ Best PD Model: {best_pd.model_name}")
    print(f"  • ROC-AUC: {best_pd.roc_auc:.4f}")
    print(f"  • Gini: {best_pd.gini:.4f}")
    print(f"  • KS Statistic: {best_pd.ks_statistic:.4f}")
    
    # Get PD predictions
    pd_predictions = best_pd.model.predict_proba(X_test_fe)[:, 1]
    
    # =========================================================================
    # STEP 4: Train LGD Models (on defaulted loans only)
    # =========================================================================
    print_header("STEP 4: Training LGD Models", "-")
    
    # Filter to defaulted loans
    default_train = y_train['default'] == 1
    default_test = y_test['default'] == 1
    
    X_train_lgd = X_train[default_train]
    y_train_lgd = y_train.loc[default_train, 'lgd']
    X_test_lgd = X_test[default_test]
    y_test_lgd = y_test.loc[default_test, 'lgd']
    
    print(f"LGD Training samples (defaulted loans): {len(X_train_lgd)}")
    
    # Re-fit feature engineer for LGD
    fe_lgd = FeatureEngineer()
    X_train_lgd_fe = fe_lgd.fit_transform(X_train_lgd)
    X_test_lgd_fe = fe_lgd.transform(X_test_lgd)
    
    lgd_results = train_all_lgd_models(
        X_train_lgd_fe, y_train_lgd,
        X_test_lgd_fe, y_test_lgd
    )
    
    print("\n" + "-" * 60)
    print("LGD Model Comparison:")
    print("-" * 60)
    print(lgd_comparison_table(lgd_results).to_string(index=False))
    
    # Select best LGD model
    best_lgd_name = min(lgd_results, key=lambda x: lgd_results[x].rmse)
    best_lgd = lgd_results[best_lgd_name]
    
    print(f"\n✓ Best LGD Model: {best_lgd.model_name}")
    print(f"  • RMSE: {best_lgd.rmse:.4f}")
    print(f"  • MAE: {best_lgd.mae:.4f}")
    print(f"  • Mean LGD: {best_lgd.mean_lgd_pred:.2%}")
    
    # =========================================================================
    # STEP 5: Calculate Expected Loss
    # =========================================================================
    print_header("STEP 5: Expected Loss Calculation", "-")
    
    # For simplicity, use mean LGD for expected loss (in practice, would predict per-loan)
    mean_lgd = best_lgd.mean_lgd_pred
    
    # Assume EAD from original data
    ead_test = X_test['loan_amount'].values if 'loan_amount' in X_test.columns else np.ones(len(X_test)) * 10000
    
    expected_loss = pd_predictions * mean_lgd * ead_test
    
    print(f"\nExpected Loss Statistics:")
    print(f"  • Average PD:  {pd_predictions.mean():.2%}")
    print(f"  • Average LGD: {mean_lgd:.2%}")
    print(f"  • Average EAD: ${ead_test.mean():,.0f}")
    print(f"  • Total Expected Loss: ${expected_loss.sum():,.0f}")
    print(f"  • Average Expected Loss per loan: ${expected_loss.mean():,.2f}")
    
    # Risk bucketing
    print("\n📊 Risk Segmentation:")
    risk_df = pd.DataFrame({
        'pd': pd_predictions,
        'expected_loss': expected_loss
    })
    risk_df['risk_bucket'] = pd.cut(
        risk_df['pd'], 
        bins=[0, 0.05, 0.15, 0.30, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    risk_summary = risk_df.groupby('risk_bucket').agg({
        'pd': ['count', 'mean'],
        'expected_loss': 'sum'
    }).round(2)
    risk_summary.columns = ['Count', 'Avg PD', 'Total EL']
    print(risk_summary)
    
    # =========================================================================
    # STEP 6: Generate Visualizations
    # =========================================================================
    print_header("STEP 6: Generating Visualizations", "-")
    
    os.makedirs('output', exist_ok=True)
    
    print("\nSaving charts to ./output/ directory...")
    
    # Chart 1: ROC Curve
    fig1 = plot_roc_curve(y_test['default'].values, pd_predictions, best_pd.model_name)
    fig1.savefig('output/roc_curve.png', dpi=150, bbox_inches='tight')
    print("  ✓ roc_curve.png")
    
    # Chart 2: KS Chart
    fig2 = plot_ks_chart(y_test['default'].values, pd_predictions)
    fig2.savefig('output/ks_chart.png', dpi=150, bbox_inches='tight')
    print("  ✓ ks_chart.png")
    
    # Chart 3: Calibration Curve
    fig3 = plot_calibration_curve(y_test['default'].values, pd_predictions, best_pd.model_name)
    fig3.savefig('output/calibration_curve.png', dpi=150, bbox_inches='tight')
    print("  ✓ calibration_curve.png")
    
    # Chart 4: Feature Importance
    fig4 = plot_feature_importance(best_pd.feature_importance)
    fig4.savefig('output/feature_importance.png', dpi=150, bbox_inches='tight')
    print("  ✓ feature_importance.png")
    
    # Chart 5: Lift Chart
    fig5 = plot_lift_chart(y_test['default'].values, pd_predictions)
    fig5.savefig('output/lift_chart.png', dpi=150, bbox_inches='tight')
    print("  ✓ lift_chart.png")
    
    # Chart 6: Model Comparison
    fig6 = plot_model_comparison(pd_results)
    fig6.savefig('output/model_comparison.png', dpi=150, bbox_inches='tight')
    print("  ✓ model_comparison.png")
    
    # Chart 7: Probability Distribution
    fig7 = plot_probability_distribution(y_test['default'].values, pd_predictions)
    fig7.savefig('output/probability_distribution.png', dpi=150, bbox_inches='tight')
    print("  ✓ probability_distribution.png")
    
    plt.close('all')
    
    # =========================================================================
    # STEP 7: Save Models
    # =========================================================================
    print_header("STEP 7: Saving Models", "-")
    
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(best_pd.model, 'models/pd_model.pkl')
    print("  ✓ Saved PD model to models/pd_model.pkl")
    
    joblib.dump(best_lgd.model, 'models/lgd_model.pkl')
    print("  ✓ Saved LGD model to models/lgd_model.pkl")
    
    joblib.dump(fe, 'models/feature_engineer.pkl')
    print("  ✓ Saved feature engineer to models/feature_engineer.pkl")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("ANALYSIS COMPLETE", "=")
    
    print("\n📊 Key Results:")
    print(f"  • PD Model: {best_pd.model_name}")
    print(f"    - ROC-AUC: {best_pd.roc_auc:.4f}")
    print(f"    - Gini: {best_pd.gini:.4f}")
    print(f"    - KS: {best_pd.ks_statistic:.4f}")
    print(f"  • LGD Model: {best_lgd.model_name}")
    print(f"    - RMSE: {best_lgd.rmse:.4f}")
    print(f"    - Mean LGD: {best_lgd.mean_lgd_pred:.2%}")
    print(f"  • Portfolio Expected Loss: ${expected_loss.sum():,.0f}")
    
    print("\n📁 Output files saved to ./output/")
    print("📁 Models saved to ./models/")
    
    print("\nDone! ✅")


if __name__ == "__main__":
    main()
