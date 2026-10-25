import numpy as np


class ConformalQuantileCalibrator:
    """
    Conformalized Quantile Regression (CQR) calibrator:
    Calibrates lower and upper quantile predictions on a held-out calibration fold
    to guarantee finite-sample nominal coverage (e.g. 90%).
    """

    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.q_correction_: float = 0.0

    def fit(
        self,
        y_true_log: np.ndarray,
        q_low_log: np.ndarray,
        q_high_log: np.ndarray,
    ) -> "ConformalQuantileCalibrator":
        """
        Compute conformal conformity scores on calibration set:
        E_i = max(q_low_i - y_i, y_i - q_high_i)
        """
        y = np.asarray(y_true_log, dtype=float)
        q_l = np.asarray(q_low_log, dtype=float)
        q_h = np.asarray(q_high_log, dtype=float)

        n = len(y)
        scores = np.maximum(q_l - y, y - q_h)

        # Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_correction_ = float(np.quantile(scores, q_level, method="higher"))
        return self

    def predict_intervals_log(
        self,
        q_low_log: np.ndarray,
        q_high_log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply calibration shift in log space."""
        calib_low = q_low_log - self.q_correction_
        calib_high = q_high_log + self.q_correction_
        return calib_low, calib_high

    def predict_intervals_eur(
        self,
        q_low_log: np.ndarray,
        q_high_log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply calibration shift and convert directly to euro space."""
        calib_low, calib_high = self.predict_intervals_log(q_low_log, q_high_log)
        low_eur = np.expm1(np.maximum(calib_low, 0.0))
        high_eur = np.expm1(np.maximum(calib_high, 0.0))
        return low_eur, high_eur
