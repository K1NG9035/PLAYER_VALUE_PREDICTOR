import os
import tempfile
from pathlib import Path

import matplotlib

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def generate_metrics_table(results_dict: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Format dictionary of model evaluation metrics into a pandas DataFrame."""
    rows = []
    for model_name, m in results_dict.items():
        rows.append({
            "Model": model_name,
            "Target": m.get("target", "—"),
            "RMSLE": round(m["rmsle"], 4),
            "MAE (€)": f"€{m['mae']:,.0f}",
            "Median AE (€)": f"€{m['median_ae']:,.0f}",
            "Within +/-25%": f"{m['within_25_pct']:.1f}%",
            "Spearman rho": round(m["spearman_rho"], 4),
        })
    return pd.DataFrame(rows)


def generate_shap_plots(
    lgbm_model,
    X_sample: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> None:
    """Generate SHAP summary and dependence plots for the LightGBM model."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_sample)

    # 1. Summary plot (Bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=15)
    plt.title("LightGBM Feature Importance (Mean |SHAP Value|)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "shap_summary_bar.png", dpi=200)
    plt.close()

    # 2. Summary plot (Beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("LightGBM SHAP Value Impact Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "shap_beeswarm.png", dpi=200)
    plt.close()

    # 3. Dependence plots for key features
    key_features = ["age", "overall", "potential_gap", "contract_years_remaining"]
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, feat in enumerate(key_features):
        if feat in X_sample.columns:
            feat_idx = list(X_sample.columns).index(feat)
            ax = axes[idx]
            ax.scatter(X_sample[feat], shap_values[:, feat_idx], alpha=0.3, color="#1f77b4", s=15)
            ax.set_title(f"SHAP Dependence: {feat}", fontsize=11, fontweight="bold")
            ax.set_xlabel(feat)
            ax.set_ylabel("SHAP value (Impact on log value)")
            ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path / "shap_dependence.png", dpi=200)
    plt.close()


def generate_error_analysis(
    test_df: pd.DataFrame,
    pred_col: str = "pred_value_eur",
    output_path: str = "reports/metrics/worst_50_misses.csv",
) -> pd.DataFrame:
    """Analyze the top 50 largest euro-space prediction misses."""
    df = test_df.copy()
    df["abs_error_eur"] = np.abs(df["value_next"] - df[pred_col])
    df["pct_error"] = (df["abs_error_eur"] / np.maximum(df["value_next"], 1.0)) * 100.0

    worst_50 = df.sort_values("abs_error_eur", ascending=False).head(50)

    cols_to_keep = [
        "sofifa_id",
        "short_name",
        "age",
        "overall",
        "potential",
        "club_name",
        "league_name",
        "position_group",
        "value_eur",
        "value_next",
        pred_col,
        "abs_error_eur",
        "pct_error",
    ]
    available_cols = [c for c in cols_to_keep if c in worst_50.columns]
    worst_50_sub = worst_50[available_cols].copy()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    worst_50_sub.to_csv(out_file, index=False)
    return worst_50_sub
