import json
import os
import tempfile
from pathlib import Path

import matplotlib

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fifa_value.data.clean import clean_panel_data
from fifa_value.data.load import load_all_seasons


def run_eda(raw_dir: str = "data/raw", report_dir: str = "reports/eda") -> dict:
    rep_path = Path(report_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    print("Loading all seasons for EDA...")
    df_raw = load_all_seasons(raw_dir=raw_dir)
    print(f"Total raw records across 7 seasons: {len(df_raw):,}")

    seasons = sorted(df_raw["season"].unique())

    # 1. Per-season overview
    summary = {
        "total_records": len(df_raw),
        "seasons": {},
        "transitions": {},
    }

    for s in seasons:
        sdf = df_raw[df_raw["season"] == s]
        zero_cnt = int((sdf["value_eur"] == 0).sum())
        non_zero_vals = sdf[sdf["value_eur"] > 0]["value_eur"]
        summary["seasons"][str(s)] = {
            "total_players": len(sdf),
            "zero_value_players": zero_cnt,
            "median_value_eur": float(non_zero_vals.median()) if len(non_zero_vals) else 0.0,
            "p90_value_eur": float(non_zero_vals.quantile(0.90)) if len(non_zero_vals) else 0.0,
            "mean_value_eur": float(sdf["value_eur"].mean()),
        }

    # 2. Club-less vs Zero Value confirmation
    p21 = df_raw[df_raw["season"] == 2021]
    p21_zero = p21[p21["value_eur"] == 0]
    p21_zero_clubless_pct = float(p21_zero["club_name"].isna().mean() * 100)
    p21_zero_wage0_pct = float((p21_zero["wage_eur"] == 0).mean() * 100)
    summary["zero_value_analysis_2021"] = {
        "total_zero_val_players": len(p21_zero),
        "pct_clubless": p21_zero_clubless_pct,
        "pct_zero_wage": p21_zero_wage0_pct,
    }

    # 3. Transitions & Attrition
    for i in range(len(seasons) - 1):
        s_curr = seasons[i]
        s_next = seasons[i + 1]
        p_curr = set(df_raw[df_raw["season"] == s_curr]["sofifa_id"])
        p_next = set(df_raw[df_raw["season"] == s_next]["sofifa_id"])
        overlap = p_curr.intersection(p_next)
        attrition = p_curr - p_next
        t_key = f"{s_curr}->{s_next}"
        summary["transitions"][t_key] = {
            "start_season_players": len(p_curr),
            "overlapping_players": len(overlap),
            "attrition_count": len(attrition),
            "attrition_rate": round(len(attrition) / len(p_curr), 4),
        }

    # Save summary json
    with open(rep_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 4. Plots: Value drift per season
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    season_years = [int(s) for s in summary["seasons"]]
    medians = [summary["seasons"][str(s)]["median_value_eur"] / 1e6 for s in season_years]
    p90s = [summary["seasons"][str(s)]["p90_value_eur"] / 1e6 for s in season_years]

    axes[0].plot(season_years, medians, marker="o", linewidth=2.5, color="#1f77b4", label="Median Value (€M)")
    axes[0].plot(season_years, p90s, marker="s", linewidth=2.5, color="#ff7f0e", label="90th Percentile (€M)")
    axes[0].set_title("EA Sports Valuation Scale Drift Across Editions", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Season (Edition)")
    axes[0].set_ylabel("Market Value (€M)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend()

    # Value by age band across 2015 vs 2021
    df_raw["age_band"] = pd.cut(
        df_raw["age"],
        bins=[0, 21, 25, 29, 100],
        labels=["<=21", "22-25", "26-29", "30+"],
    )
    drift_age = df_raw.groupby(["season", "age_band"], observed=True)["value_eur"].median().unstack() / 1e6
    drift_age.plot(ax=axes[1], marker="o", colormap="viridis")
    axes[1].set_title("Median Player Value by Age Band Over Time (€M)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Median Value (€M)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(rep_path / "value_drift.png", dpi=200)
    plt.close()

    # 5. Attrition plot
    _fig, ax = plt.subplots(figsize=(8, 5))
    trans_labels = list(summary["transitions"].keys())
    attrition_rates = [summary["transitions"][k]["attrition_rate"] * 100 for k in trans_labels]
    bars = ax.bar(trans_labels, attrition_rates, color="#d62728", alpha=0.85, edgecolor="black")
    ax.set_title("Player Attrition Rate Between Consecutive Seasons (%)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Attrition Rate (%)")
    ax.set_ylim(0, 35)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.8, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(rep_path / "attrition_analysis.png", dpi=200)
    plt.close()

    # 6. History length distribution at 2021
    clean_df = clean_panel_data(df_raw)
    p21_clean = clean_df[clean_df["season"] == 2021]
    history_counts = clean_df.groupby("sofifa_id")["season"].count()
    p21_hist_lens = history_counts.loc[p21_clean["sofifa_id"]].value_counts().sort_index()

    _fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(p21_hist_lens.index, p21_hist_lens.values, color="#2ca02c", alpha=0.85, edgecolor="black")
    ax.set_title("History Length Distribution of Players in FIFA 21", fontsize=12, fontweight="bold")
    ax.set_xlabel("Seasons Observed Prior to / At FIFA 21")
    ax.set_ylabel("Number of Players")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 100, f"{yval:,}", ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(rep_path / "history_length_dist.png", dpi=200)
    plt.close()

    print("EDA completed successfully. Reports saved to", rep_path)
    return summary


if __name__ == "__main__":
    run_eda()
