# Model Card: FIFA Player Value Forecaster (Hybrid Ensemble)

## 1. Model Details
- **Model Name:** FIFA Player Market Value Forecaster & Calibrated Uncertainty Estimator
- **Model Type:** Hybrid Ensemble (LightGBM + CatBoost + Sequence GRU) with Non-Negative Linear Stacking and Conformalized Quantile Regression (CQR).
- **Version:** 1.0.0
- **Developers:** Senior Machine Learning Team, EA Sports Analytics
- **Primary Release Date:** October 2026

## 2. Intended Use
- **Primary Use Case:** Forecasting in-game player market valuations (`value_eur`) for season $t+1$ (specifically forecasting FIFA 22 values using FIFA 21 and earlier historical data), alongside calibrated 90% prediction intervals.
- **Applications:** Career Mode transfer balance, squad-building economy tuning, automated player valuation sanity checks, and identifying high-uncertainty young breakout prospects vs declining veteran assets.
- **Out-of-Scope Uses:** Real-world transfer fee estimations, legal/contractual financial negotiations, or player scouting in non-FIFA professional football clubs.

## 3. Data & Training Factors
- **Input Data:** EA Sports FIFA Career Mode datasets across seven seasons (`players_15.csv` through `players_21.csv`).
- **Target Formulations:**
  - **Target A (Level):** $\log(1 + \text{value}_{t+1})$
  - **Target B (Growth):** $\log(1 + \text{value}_{t+1}) - \log(1 + \text{value}_t)$
- **Temporal Split Scheme:**
  - Expanding chronological window validation across 3 folds:
    - Fold 1: Train 15->16, 16->17 | Validate 17->18
    - Fold 2: Train 15->16 ... 17->18 | Validate 18->19
    - Fold 3: Train 15->16 ... 18->19 | Validate 19->20
  - Held-out Test Transition: 2020->2021 (touched strictly once for final evaluation).
  - Out-of-Sample Inference: Forecasting 2022 from 2021 records.

## 4. Evaluation Metrics
Evaluations are computed in original euro space (`expm1`) across an identical set of non-zero valuation rows:
- **RMSLE:** Root Mean Squared Logarithmic Error (Scale-free primary ranking metric).
- **MAE & Median AE (€):** Absolute valuation errors across stars vs squad players.
- **Percentage Accuracy Bands:** Percentage of predictions within $\pm 10\%$ and $\pm 25\%$ of true value.
- **Spearman Rank Correlation ($\rho$):** Rank-ordering consistency.
- **Prediction Interval Coverage & Winkler Score:** Empirical 90% prediction interval coverage and interval penalty score.

## 5. Limitations & Biases
1. **EA Valuation vs Real Transfer Market:** In-game `value_eur` is an algorithmic valuation produced by EA game designers based on player attributes, potential, age, reputation, and international status. It does not reflect real-life transfer bids, bidding wars, agent fees, release clause clauses, or club financial distress.
2. **Survivorship & Attrition Bias:** Players who retire, transfer to unlicensed leagues, or are omitted from season $t+1$ have no ground truth label. The model conditions predictions on the player remaining active in FIFA.
3. **Scale Drift:** EA's internal currency calibration shifts across editions. Features are normalized by per-season percentiles and context aggregates to avoid tree extrapolation errors.
4. **Administrative / Zero-Value Entries:** Players with €0 value (typically clubless or unlicensed national team placeholders) are excluded from target modeling and flagged as inputs.
5. **History Truncation:** Most players appear in only 1 to 3 seasons; historical trajectory slopes are imputed or masked accordingly.
