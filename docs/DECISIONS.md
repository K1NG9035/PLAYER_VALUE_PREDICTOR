# Architecture and Modeling Decisions Log

This document records the architectural and design decisions made for the FIFA Player Value Forecasting system, including the context, rationale, and consequences of each decision.

---

### DECISION 001: Log-Space Modeling and Euro-Space Evaluation Metric Parity
- **Context:** Player market values span from €0 to €105.5M+ and are heavily right-skewed. Tree-based splits and gradient descent in raw euro space would focus almost entirely on the top 1% of superstars, ignoring 99% of squad players.
- **Decision:** Model all targets in log space (`log1p(value_eur)` for Level / Target A, and `log1p(value_{t+1}) - log1p(value_t)` for Growth / Target B). For evaluation, convert all predictions back to euros via `expm1` and compute RMSLE, MAE, Median AE, and percentage accuracy bands on the exact same row set.
- **Rationale:** Log-space modeling stabilizes variance and penalizes proportional errors equally across tiers. Scored evaluation in common euro space guarantees an honest head-to-head comparison between Target A and Target B.

---

### DECISION 002: Temporal Validation with Expanding Chronological Windows
- **Context:** Over 4,000 players appear in all seven seasons, and ~12,000+ appear in consecutive seasons. Random train/test splits would result in extreme temporal leakage (e.g. knowing a player's rating in 2019 to predict 2018).
- **Decision:** Implement a strictly chronological expanding-window temporal CV:
  - Fold 1: Train [15->16, 16->17], Val [17->18]
  - Fold 2: Train [15->16, 16->17, 17->18], Val [18->19]
  - Fold 3: Train [15->16, 16->17, 17->18, 18->19], Val [19->20]
  - Final Test: Train [15->16 ... 19->20], Calibration [19->20], Test [20->21] (touched strictly once).
- **Rationale:** Mirrors the real-world operational setting where only historical seasons are observed, preventing lookahead leakage.

---

### DECISION 003: Exclusion and Flagging of Zero-Value Players
- **Context:** Across seasons, 230–310 players have `value_eur == 0`. Analysis indicates >95% of these players are club-less, have `wage_eur == 0`, and often carry placeholder names for unlicensed federations.
- **Decision:** Exclude transitions where either `value_t == 0` or `value_{t+1} == 0` from the primary training and evaluation target sets. Add an explicit binary feature `is_clubless` and `value_is_zero_t` as input features, and provide a sensitivity analysis including zero-value rows.
- **Rationale:** Zero values represent administrative/licensing artifacts rather than economic market value. Including them in log-growth targets would create infinite or artificial log shifts (`log(0)`).

---

### DECISION 004: Anti-Scale Drift Feature Normalization
- **Context:** EA's internal valuation formula drifts between annual editions (e.g., median value jumps from €350k in FIFA 15 to €725k in FIFA 20). Decision trees cannot extrapolate when fed the raw `season` integer.
- **Decision:** Never pass raw `season` as a numerical or categorical feature to tree models. Instead, construct per-season percentile ranks (e.g., `overall_season_rank`, `value_season_rank`), and relative context features (deviation from season league median and position median).
- **Rationale:** Per-season relative metrics are stationary across time and generalize safely to unseen future seasons such as FIFA 22.

---

### DECISION 005: Clean Backward-Only Trajectory Aggregation
- **Context:** To capture player career arcs (improving youngsters vs declining veterans), multi-season lag features (lags 1–3, deltas, rolling means, and historical trajectory slopes) are required.
- **Decision:** For any pair at season $t$, calculate all trajectory metrics strictly using observations from seasons $s \le t$. History gaps are handled natively as NaNs in tree models and with masking in the sequence branch.
- **Rationale:** Strictly enforces causality. A dedicated unit test asserts that modifying season $t+1$ features produces zero change in the feature matrix for $(p, t)$.

---

### DECISION 006: Hybrid Ensemble with Stacking and Conformal Calibration
- **Context:** Tabular GBDT models excel on dense attribute vectors, while sequence models (GRU) capture temporal dynamics across variable-length history. Point predictions also do not communicate downside risk.
- **Decision:** Combine LightGBM and CatBoost with a PyTorch GRU using non-negative stacking on temporal out-of-fold predictions. Compute prediction intervals using multi-quantile heads (0.05, 0.50, 0.95) coupled with Conformalized Quantile Regression (CQR) calibrated on 2019->2020.
- **Rationale:** Stacking exploits complementary model representations without overfitting, while CQR provides finite-sample coverage guarantees targeting 90% empirical coverage on unseen transitions.
