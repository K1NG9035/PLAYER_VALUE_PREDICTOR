import numpy as np
from scipy.stats import spearmanr


def compute_regression_metrics(
    y_true_eur: np.ndarray,
    y_pred_eur: np.ndarray,
    floor_eur: float = 1000.0,
) -> dict[str, float]:
    """
    Compute headline evaluation metrics in euro space.
    y_true_eur: actual market value in euros
    y_pred_eur: predicted market value in euros
    """
    y_true = np.asarray(y_true_eur, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred_eur, dtype=float), 0.0)

    # Errors
    abs_errors = np.abs(y_true - y_pred)
    mae = float(np.mean(abs_errors))
    median_ae = float(np.median(abs_errors))

    # Log space RMSE (RMSLE)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    rmsle = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))

    # Relative percentage errors (floored to avoid div-by-zero on small values)
    floored_true = np.maximum(y_true, floor_eur)
    pct_errors = abs_errors / floored_true
    mape = float(np.mean(pct_errors) * 100.0)
    smape = float(np.mean(2.0 * abs_errors / (y_true + y_pred + 1e-6)) * 100.0)

    # Accuracy bands
    within_10 = float(np.mean(pct_errors <= 0.10) * 100.0)
    within_25 = float(np.mean(pct_errors <= 0.25) * 100.0)

    # Spearman rank correlation
    if len(np.unique(y_pred)) > 1:
        rho, _ = spearmanr(y_true, y_pred)
        spearman_corr = float(rho)
    else:
        spearman_corr = 0.0

    return {
        "rmsle": rmsle,
        "mae": mae,
        "median_ae": median_ae,
        "mape": mape,
        "smape": smape,
        "within_10_pct": within_10,
        "within_25_pct": within_25,
        "spearman_rho": spearman_corr,
    }


def compute_interval_metrics(
    y_true_eur: np.ndarray,
    lower_eur: np.ndarray,
    upper_eur: np.ndarray,
    nominal_coverage: float = 0.90,
) -> dict[str, float]:
    """
    Compute prediction interval metrics:
    - empirical coverage (%)
    - mean width in euros and log space
    - Winkler interval score
    """
    y_true = np.asarray(y_true_eur, dtype=float)
    lower = np.asarray(lower_eur, dtype=float)
    upper = np.asarray(upper_eur, dtype=float)

    alpha = 1.0 - nominal_coverage

    # Coverage
    in_interval = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(in_interval))

    # Width
    widths_eur = upper - lower
    mean_width_eur = float(np.mean(widths_eur))
    median_width_eur = float(np.median(widths_eur))

    log_upper = np.log1p(np.maximum(upper, 0.0))
    log_lower = np.log1p(np.maximum(lower, 0.0))
    mean_width_log = float(np.mean(log_upper - log_lower))

    # Winkler Score: width + penalty if outside
    # S(l, u, y) = (u - l) + (2/alpha)*(l - y) if y < l + (2/alpha)*(y - u) if y > u
    penalty_lower = np.where(y_true < lower, (2.0 / alpha) * (lower - y_true), 0.0)
    penalty_upper = np.where(y_true > upper, (2.0 / alpha) * (y_true - upper), 0.0)
    winkler_eur = widths_eur + penalty_lower + penalty_upper
    mean_winkler = float(np.mean(winkler_eur))

    # Log-space Winkler score
    log_y = np.log1p(y_true)
    log_w = log_upper - log_lower
    pen_l_log = np.where(log_y < log_lower, (2.0 / alpha) * (log_lower - log_y), 0.0)
    pen_u_log = np.where(log_y > log_upper, (2.0 / alpha) * (log_y - log_upper), 0.0)
    winkler_log = float(np.mean(log_w + pen_l_log + pen_u_log))

    return {
        "coverage": coverage,
        "mean_width_eur": mean_width_eur,
        "median_width_eur": median_width_eur,
        "mean_width_log": mean_width_log,
        "winkler_score_eur": mean_winkler,
        "winkler_score_log": winkler_log,
    }


def paired_bootstrap_ci(
    y_true_eur: np.ndarray,
    preds_a_eur: np.ndarray,
    preds_b_eur: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute paired bootstrap difference in RMSLE: RMSLE(A) - RMSLE(B).
    Returns (diff_mean, ci_lower, ci_upper).
    Negative diff means Model A has lower RMSLE (A is better).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true_eur)
    log_true = np.log1p(y_true_eur)
    log_a = np.log1p(preds_a_eur)
    log_b = np.log1p(preds_b_eur)

    diffs = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        rmsle_a = np.sqrt(np.mean((log_true[idx] - log_a[idx]) ** 2))
        rmsle_b = np.sqrt(np.mean((log_true[idx] - log_b[idx]) ** 2))
        diffs.append(rmsle_a - rmsle_b)

    diffs = np.array(diffs)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(diffs, alpha * 100.0))
    ci_upper = float(np.percentile(diffs, (1.0 - alpha) * 100.0))
    return float(np.mean(diffs)), ci_lower, ci_upper
