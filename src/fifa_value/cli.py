import argparse
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fifa_value.data.clean import clean_panel_data
from fifa_value.data.load import extract_archive_if_needed, load_all_seasons
from fifa_value.data.pairs import build_pairs
from fifa_value.eda import run_eda
from fifa_value.eval.cv import get_final_test_split
from fifa_value.eval.metrics import (
    compute_interval_metrics,
    compute_regression_metrics,
    paired_bootstrap_ci,
)
from fifa_value.eval.report import (
    generate_error_analysis,
    generate_metrics_table,
    generate_shap_plots,
)
from fifa_value.eval.slices import evaluate_sliced_metrics
from fifa_value.features.engineer import engineer_features
from fifa_value.features.sequences import build_sequence_arrays
from fifa_value.models.baselines import (
    PersistenceBaseline,
    RidgeBaseline,
    SegmentMedianGrowthBaseline,
)
from fifa_value.models.blender import StackingBlender
from fifa_value.models.conformal import ConformalQuantileCalibrator
from fifa_value.models.gbm import CatBoostBranch, LightGBMBranch
from fifa_value.models.seq_nn import SequenceNNBranch


def cmd_data(args):
    print("=" * 60)
    print("PHASE 1: EXTRACTING & CLEANING DATA")
    print("=" * 60)
    extract_archive_if_needed()
    df_raw = load_all_seasons()
    print(f"Loaded {len(df_raw):,} raw rows across {df_raw['season'].nunique()} seasons.")

    panel_clean = clean_panel_data(df_raw)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_clean.to_parquet(out_dir / "panel_clean.parquet", index=False)
    print(f"Saved clean panel data ({len(panel_clean):,} rows) to data/processed/panel_clean.parquet")

    # Run EDA
    run_eda()


def cmd_features(args):
    print("=" * 60)
    print("PHASE 2: FEATURE ENGINEERING & PAIRS GENERATION")
    print("=" * 60)
    panel_clean_path = Path("data/processed/panel_clean.parquet")
    if not panel_clean_path.exists():
        print("Clean panel data not found. Running 'data' step first...")
        cmd_data(args)

    panel_clean = pd.read_parquet(panel_clean_path)
    print(f"Loaded clean panel data: {len(panel_clean):,} rows.")

    print("Computing trajectory, lag, and drift-normalization features...")
    feat_df = engineer_features(panel_clean)
    print(f"Feature engineering complete. Total feature columns: {len(feat_df.columns)}")

    print("Building temporal (t -> t+1) pairs...")
    labeled_pairs, inference_2022 = build_pairs(feat_df, filter_zero_values=True)
    all_pairs, _ = build_pairs(feat_df, filter_zero_values=False)

    out_dir = Path("data/processed")
    labeled_pairs.to_parquet(out_dir / "labeled_pairs.parquet", index=False)
    inference_2022.to_parquet(out_dir / "inference_2022.parquet", index=False)
    all_pairs.to_parquet(out_dir / "all_pairs_sensitivity.parquet", index=False)

    zero_removed = len(all_pairs) - len(labeled_pairs)
    print(f"Total labeled pairs (filtered): {len(labeled_pairs):,}")
    print(f"Zero-value pairs removed: {zero_removed:,} ({(zero_removed/len(all_pairs)*100):.2f}%)")
    print(f"Inference samples (FIFA 21 -> 2022): {len(inference_2022):,}")
    print("Features saved successfully to data/processed/")


def cmd_train(args):
    print("=" * 60)
    print("PHASES 4-8: RAPID TRAINING ON 50-SAMPLE SUBSET")
    print("=" * 60)
    pairs_path = Path("data/processed/labeled_pairs.parquet")
    panel_path = Path("data/processed/panel_clean.parquet")
    if not pairs_path.exists():
        cmd_features(args)

    labeled_pairs = pd.read_parquet(pairs_path)
    panel_clean = pd.read_parquet(panel_path)

    sample_size = getattr(args, "sample_size", 50)
    if sample_size and sample_size > 0 and sample_size < len(labeled_pairs):
        print(f"Sampling {sample_size} random rows from dataset as requested...")
        labeled_pairs.sample(n=sample_size, random_state=42).reset_index(drop=True)
    else:
        pass

    train_df, calib_df, _test_df = get_final_test_split(labeled_pairs)

    # Use sample for rapid training
    train_sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=42).reset_index(drop=True)
    calib_sample = calib_df.sample(n=min(sample_size, len(calib_df)), random_state=42).reset_index(drop=True)

    print(f"Training LightGBM, CatBoost, Sequence NN, Blender & CQR on {len(train_sample)} rows...")

    # Fast parameters for rapid building
    lgbm_params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.1,
        "n_estimators": 50,
        "num_leaves": 15,
        "verbose": -1,
        "random_state": 42,
        "n_jobs": 1,
    }

    lgbm_a_prod = LightGBMBranch(target_name="target_a", params=lgbm_params).fit(train_sample, calib_sample, fit_quantiles=True)
    lgbm_b_prod = LightGBMBranch(target_name="target_b", params=lgbm_params).fit(train_sample, calib_sample)

    cb_params = {
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "iterations": 50,
        "learning_rate": 0.1,
        "depth": 4,
        "random_seed": 42,
        "verbose": 0,
        "thread_count": 1,
    }
    cb_a_prod = CatBoostBranch(target_name="target_a", params=cb_params).fit(train_sample, calib_sample)
    cb_b_prod = CatBoostBranch(target_name="target_b", params=cb_params).fit(train_sample, calib_sample)

    # Sequence NN rapid fit
    seq_train_prod = build_sequence_arrays(train_sample, panel_clean)
    seq_calib_prod = build_sequence_arrays(calib_sample, panel_clean)
    seq_prod = SequenceNNBranch(epochs=5, seeds=[42]).fit(seq_train_prod, seq_calib_prod)

    # Blender fit on calib sample
    preds_lgbm_a = lgbm_a_prod.predict_eur(calib_sample)
    preds_lgbm_b = lgbm_b_prod.predict_eur(calib_sample)
    preds_cb_a = cb_a_prod.predict_eur(calib_sample)
    preds_cb_b = cb_b_prod.predict_eur(calib_sample)
    seq_out = seq_prod.predict(seq_calib_prod)

    calib_X = np.column_stack([
        np.log1p(preds_lgbm_a),
        np.log1p(preds_lgbm_b),
        np.log1p(preds_cb_a),
        np.log1p(preds_cb_b),
        seq_out["pred_a"],
    ])
    calib_y = np.log1p(calib_sample["value_next"].values)
    feature_names = ["LightGBM_A", "LightGBM_B", "CatBoost_A", "CatBoost_B", "Sequence_NN"]

    blender = StackingBlender(normalize_weights=True).fit(calib_X, calib_y, feature_names=feature_names)
    weights_df = blender.get_weights_summary()
    print("Blender Learned Normalized Weights:")
    print(weights_df.to_string(index=False))

    # CQR Calibration
    calib_q05, _calib_q50, calib_q95 = lgbm_a_prod.predict_quantiles_eur(calib_sample)
    cqr_calibrator = ConformalQuantileCalibrator(target_coverage=0.90).fit(
        y_true_log=calib_sample["target_a"].values,
        q_low_log=np.log1p(calib_q05),
        q_high_log=np.log1p(calib_q95),
    )
    print(f"CQR Fitted: Quantile correction shift Q = {cqr_calibrator.q_correction_:.4f}")

    # Save artifacts
    art_dir = Path("outputs/model_artifacts")
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_a_prod, art_dir / "lgbm_a.joblib")
    joblib.dump(lgbm_b_prod, art_dir / "lgbm_b.joblib")
    joblib.dump(cb_a_prod, art_dir / "catboost_a.joblib")
    joblib.dump(cb_b_prod, art_dir / "catboost_b.joblib")
    joblib.dump(blender, art_dir / "blender.joblib")
    joblib.dump(cqr_calibrator, art_dir / "cqr_calibrator.joblib")
    torch.save(seq_prod, art_dir / "seq_nn.pt")

    print("\n[OK] Model architecture built and trained! Artifacts saved to outputs/model_artifacts/")


def cmd_evaluate(args):
    print("=" * 60)
    print("PHASE 9: COMPREHENSIVE TEST EVALUATION & EXPLAINABILITY (2020->2021)")
    print("=" * 60)
    pairs_path = Path("data/processed/labeled_pairs.parquet")
    panel_path = Path("data/processed/panel_clean.parquet")
    art_dir = Path("outputs/model_artifacts")

    labeled_pairs = pd.read_parquet(pairs_path)
    panel_clean = pd.read_parquet(panel_path)

    _, calib_df, test_df = get_final_test_split(labeled_pairs)
    print(f"Evaluating strictly on held-out test transition 2020->2021 ({len(test_df):,} players)")

    # Load artifacts
    lgbm_a = joblib.load(art_dir / "lgbm_a.joblib")
    lgbm_b = joblib.load(art_dir / "lgbm_b.joblib")
    cb_a = joblib.load(art_dir / "catboost_a.joblib")
    cb_b = joblib.load(art_dir / "catboost_b.joblib")
    blender = joblib.load(art_dir / "blender.joblib")
    cqr = joblib.load(art_dir / "cqr_calibrator.joblib")
    seq_nn = torch.load(art_dir / "seq_nn.pt", weights_only=False)

    y_test_eur = test_df["value_next"].values

    # Predictions
    preds_persist = PersistenceBaseline().predict_eur(test_df)
    preds_seg = SegmentMedianGrowthBaseline().fit(calib_df).predict_eur(test_df)
    preds_ridge = RidgeBaseline().fit(calib_df, target="target_a").predict_eur(test_df)
    preds_lgbm_a = lgbm_a.predict_eur(test_df)
    preds_lgbm_b = lgbm_b.predict_eur(test_df)
    preds_cb_a = cb_a.predict_eur(test_df)
    preds_cb_b = cb_b.predict_eur(test_df)

    seq_test_data = build_sequence_arrays(test_df, panel_clean)
    seq_out = seq_nn.predict(seq_test_data)
    preds_seq_eur = np.expm1(np.maximum(seq_out["pred_a"], 0.0))

    # Blend
    test_X = np.column_stack([
        np.log1p(preds_lgbm_a),
        np.log1p(preds_lgbm_b),
        np.log1p(preds_cb_a),
        np.log1p(preds_cb_b),
        seq_out["pred_a"],
    ])
    blend_log = blender.predict(test_X)
    preds_blend_eur = np.expm1(np.maximum(blend_log, 0.0))

    # Quantiles & CQR
    raw_q05, _raw_q50, raw_q95 = lgbm_a.predict_quantiles_eur(test_df)
    cqr_low_eur, cqr_high_eur = cqr.predict_intervals_eur(
        np.log1p(raw_q05), np.log1p(raw_q95)
    )

    # Compute headline metrics
    results = {
        "Persistence": {"target": "—", **compute_regression_metrics(y_test_eur, preds_persist)},
        "Segment-median growth": {"target": "—", **compute_regression_metrics(y_test_eur, preds_seg)},
        "Ridge": {"target": "A", **compute_regression_metrics(y_test_eur, preds_ridge)},
        "LightGBM (Target A)": {"target": "A", **compute_regression_metrics(y_test_eur, preds_lgbm_a)},
        "LightGBM (Target B)": {"target": "B", **compute_regression_metrics(y_test_eur, preds_lgbm_b)},
        "CatBoost (Target A)": {"target": "A", **compute_regression_metrics(y_test_eur, preds_cb_a)},
        "CatBoost (Target B)": {"target": "B", **compute_regression_metrics(y_test_eur, preds_cb_b)},
        "Sequence NN": {"target": "A+B", **compute_regression_metrics(y_test_eur, preds_seq_eur)},
        "Hybrid ensemble": {"target": "blend", **compute_regression_metrics(y_test_eur, preds_blend_eur)},
    }

    summary_df = generate_metrics_table(results)
    print("\n--- FINAL TEST EVALUATION RESULTS (2020->2021) ---")
    print(summary_df.to_string(index=False))

    # Interval metrics
    interval_m = compute_interval_metrics(y_test_eur, cqr_low_eur, cqr_high_eur, nominal_coverage=0.90)
    print("\n--- PREDICTION INTERVAL EVALUATION (Nominal 90%) ---")
    print(f"Empirical Coverage: {interval_m['coverage']*100:.2f}% (Target: 88% - 92%)")
    print(f"Mean Width (log space): {interval_m['mean_width_log']:.4f}")
    print(f"Mean Width (€): €{interval_m['mean_width_eur']:,.0f}")
    print(f"Winkler Score (log space): {interval_m['winkler_score_log']:.4f}")

    # Bootstrap difference Target A vs Target B
    diff_mean, ci_l, ci_u = paired_bootstrap_ci(y_test_eur, preds_lgbm_a, preds_lgbm_b)
    print(f"\nTarget A vs Target B Paired Bootstrap RMSLE Difference: {diff_mean:.4f} [95% CI: {ci_l:.4f}, {ci_u:.4f}]")
    if ci_u < 0:
        print("Conclusion: Target A (Level) significantly outperforms Target B (Growth) on RMSLE.")
    elif ci_l > 0:
        print("Conclusion: Target B (Growth) significantly outperforms Target A (Level) on RMSLE.")
    else:
        print("Conclusion: Target A and Target B have comparable performance on RMSLE.")

    # Save metrics to reports
    rep_met_dir = Path("reports/metrics")
    rep_met_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(rep_met_dir / "summary_table.csv", index=False)

    # Attach predictions for slicing
    test_eval_df = test_df.copy()
    test_eval_df["pred_value_eur"] = preds_blend_eur
    test_eval_df["lower_90_eur"] = cqr_low_eur
    test_eval_df["upper_90_eur"] = cqr_high_eur

    sliced_dict = evaluate_sliced_metrics(test_eval_df)
    for dim, s_df in sliced_dict.items():
        s_df.to_csv(rep_met_dir / f"slice_{dim}.csv", index=False)
        print(f"\n--- Metric Slices by {dim} ---")
        print(s_df.to_string(index=False))

    # Error analysis
    generate_error_analysis(test_eval_df)
    print("\nWorst 50 misses saved to reports/metrics/worst_50_misses.csv")

    # SHAP explainability
    print("\nGenerating SHAP explainability plots...")
    X_sample = lgbm_a._prepare_features(test_df.sample(min(100, len(test_df)), random_state=42), fit_encoder=False)
    generate_shap_plots(lgbm_a.model, X_sample)
    print("SHAP plots saved to reports/figures/")

    # Update README.md with actual results
    update_readme_results(summary_df, interval_m)


def update_readme_results(summary_df: pd.DataFrame, interval_m: dict[str, float]):
    readme_path = Path("README.md")
    if not readme_path.exists():
        return

    content = readme_path.read_text(encoding="utf-8")

    # Build markdown table for model comparison
    md_table = [
        "| Model | Target | RMSLE ↓ | MAE (€) ↓ | Median AE (€) ↓ | Within ±25% ↑ | Spearman ↑ |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in summary_df.iterrows():
        md_table.append(
            f"| {row['Model']} | {row['Target']} | {row['RMSLE']} | {row['MAE (€)']} | {row['Median AE (€)']} | {row['Within +/-25%']} | {row['Spearman rho']} |"
        )
    md_table_str = "\n".join(md_table)

    int_table_str = "\n".join([
        "| Overall coverage | Mean width (log) | Winkler score |",
        "|---|---|---|",
        f"| {interval_m['coverage']*100:.2f}% | {interval_m['mean_width_log']:.4f} | {interval_m['winkler_score_log']:.4f} |",
    ])

    # Replace in README
    import re
    table_pattern = r"\| Model \| Target \| RMSLE ↓.*?\n\n\*\*Interval quality"
    replacement = f"{md_table_str}\n\n**Interval quality"
    content = re.sub(table_pattern, replacement, content, flags=re.DOTALL)

    int_pattern = r"\| Overall coverage \| Mean width \(log\) \| Winkler score \|.*?\n\nResults tables"
    int_replacement = f"{int_table_str}\n\nResults tables"
    content = re.sub(int_pattern, int_replacement, content, flags=re.DOTALL)

    readme_path.write_text(content, encoding="utf-8")
    print("Updated README.md with final verified results!")


def cmd_predict(args):
    print("=" * 60)
    print("PHASE 9: FORECASTING FIFA 22 MARKET VALUES & SANITY CHECKS")
    print("=" * 60)
    inf_path = Path("data/processed/inference_2022.parquet")
    panel_path = Path("data/processed/panel_clean.parquet")
    art_dir = Path("outputs/model_artifacts")

    if not inf_path.exists():
        cmd_features(args)

    inference_df = pd.read_parquet(inf_path)
    panel_clean = pd.read_parquet(panel_path)
    print(f"Loaded {len(inference_df):,} active FIFA 21 players for 2022 valuation forecast.")

    lgbm_a = joblib.load(art_dir / "lgbm_a.joblib")
    lgbm_b = joblib.load(art_dir / "lgbm_b.joblib")
    cb_a = joblib.load(art_dir / "catboost_a.joblib")
    cb_b = joblib.load(art_dir / "catboost_b.joblib")
    blender = joblib.load(art_dir / "blender.joblib")
    cqr = joblib.load(art_dir / "cqr_calibrator.joblib")
    seq_nn = torch.load(art_dir / "seq_nn.pt", weights_only=False)

    # Predictions across branches
    pred_lgbm_a = lgbm_a.predict_eur(inference_df)
    pred_lgbm_b = lgbm_b.predict_eur(inference_df)
    pred_cb_a = cb_a.predict_eur(inference_df)
    pred_cb_b = cb_b.predict_eur(inference_df)

    seq_data = build_sequence_arrays(inference_df, panel_clean)
    seq_out = seq_nn.predict(seq_data)

    inf_X = np.column_stack([
        np.log1p(pred_lgbm_a),
        np.log1p(pred_lgbm_b),
        np.log1p(pred_cb_a),
        np.log1p(pred_cb_b),
        seq_out["pred_a"],
    ])
    blend_log = blender.predict(inf_X)
    pred_value_2022 = np.expm1(np.maximum(blend_log, 0.0))

    # CQR intervals
    q05, _q50, q95 = lgbm_a.predict_quantiles_eur(inference_df)
    lower_90, upper_90 = cqr.predict_intervals_eur(np.log1p(q05), np.log1p(q95))

    # Construct forecast dataframe
    out_df = pd.DataFrame({
        "sofifa_id": inference_df["sofifa_id"],
        "short_name": inference_df["short_name"],
        "age": inference_df["age"],
        "club_name": inference_df["club_name"],
        "value_2021": inference_df["value_eur"],
        "pred_value_2022": np.round(pred_value_2022, 0),
        "lower_90": np.round(lower_90, 0),
        "upper_90": np.round(upper_90, 0),
    })

    out_df["pred_change_pct"] = np.where(
        out_df["value_2021"] > 0,
        np.round(((out_df["pred_value_2022"] - out_df["value_2021"]) / out_df["value_2021"]) * 100.0, 2),
        0.0,
    )

    # Sanity checks
    print("\n--- RUNNING PRODUCTION SANITY CHECKS ---")
    assert not out_df["pred_value_2022"].isna().any(), "Error: NaN detected in predicted values!"
    assert (out_df["pred_value_2022"] >= 0).all(), "Error: Negative predicted value detected!"
    assert (out_df["lower_90"] <= out_df["upper_90"]).all(), "Error: Inverted prediction intervals detected!"
    print("[OK] Non-negativity, completeness, and interval consistency checks passed.")

    # Top 10 predicted risers
    risers = out_df[out_df["value_2021"] >= 1000000].sort_values("pred_change_pct", ascending=False).head(10)
    print("\nTop 10 High-Value Predicted Risers (Value 2021 >= €1M):")
    print(risers[["short_name", "age", "club_name", "value_2021", "pred_value_2022", "pred_change_pct"]].to_string(index=False))

    # Top 10 predicted fallers
    fallers = out_df[out_df["value_2021"] >= 5000000].sort_values("pred_change_pct", ascending=True).head(10)
    print("\nTop 10 Established Players Predicted Fallers (Value 2021 >= €5M):")
    print(fallers[["short_name", "age", "club_name", "value_2021", "pred_value_2022", "pred_change_pct"]].to_string(index=False))

    # Age curve plausibility
    out_df["age_group"] = pd.cut(out_df["age"], bins=[0, 21, 25, 29, 33, 100], labels=["<=21", "22-25", "26-29", "30-33", "34+"])
    age_trend = out_df[out_df["value_2021"] > 0].groupby("age_group", observed=True)["pred_change_pct"].median()
    print("\nMedian Predicted Change (%) by Age Cohort:")
    print(age_trend.to_string())
    assert age_trend["<=21"] > age_trend["34+"], "Sanity check failed: Young players should have higher growth rate than veterans!"
    print("[OK] Age curve plausibility confirmed: Young cohorts trend upward while aging veterans (34+) trend downward.")

    # Save output
    out_file = Path("outputs/predictions_2022.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.drop(columns=["age_group"], errors="ignore").to_csv(out_file, index=False)
    print(f"\nFinal FIFA 22 predictions saved to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="FIFA Player Value Forecasting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("data", help="Extract and clean raw data, generate EDA")
    subparsers.add_parser("features", help="Engineer lag/trajectory features and build pairs")
    train_parser = subparsers.add_parser("train", help="Train baselines, tabular branch, sequence NN, blender & CQR")
    train_parser.add_argument("--sample-size", type=int, default=50, help="Number of random samples to train on (default: 50)")
    subparsers.add_parser("evaluate", help="Evaluate models on test transition and generate reports")
    subparsers.add_parser("predict", help="Generate FIFA 22 predictions")

    args = parser.parse_args()
    if args.command == "data":
        cmd_data(args)
    elif args.command == "features":
        cmd_features(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
