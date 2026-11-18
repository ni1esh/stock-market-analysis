"""
=============================================================
  NIFTY 50 NEXT-DAY OPENING PRICE PREDICTION — FULL PIPELINE
=============================================================
  Steps:
    1. Load & clean raw OHLCV data
    2. Engineer technical indicator features
    3. Build stationary (scale-free) features
    4. Train/Val/Test chronological split
    5. Baseline Random Forest (all features)
    6. RFE feature selection (tuned on validation)
    7. Final RFE-RF model on test set
    8. Evaluate: MAE, RMSE, R², Directional Accuracy
    9. Save results, plots, and selected features
=============================================================
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_CSV         = os.path.join(BASE_DIR, "src", "data", "raw",       "NIFTY50_clean.csv")
FEATURES_CSV    = os.path.join(BASE_DIR, "src", "data", "processed", "NIFTY50_features.csv")
RESULTS_DIR     = os.path.join(BASE_DIR, "src", "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
METRICS_CSV     = os.path.join(RESULTS_DIR, "metrics_summary.csv")
FEATURES_JSON   = os.path.join(RESULTS_DIR, "selected_features.json")

os.makedirs(PLOTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def step(n, title):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  STEP {n}: {title}")
    print(f"{bar}")

def evaluate(y_true, y_pred, label=""):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)

    # Directional Accuracy — did we get up/down right?
    actual_dir    = np.sign(np.array(y_true).flatten()[1:] - np.array(y_true).flatten()[:-1])
    predicted_dir = np.sign(np.array(y_pred).flatten()[1:] - np.array(y_pred).flatten()[:-1])
    da = np.mean(actual_dir == predicted_dir) * 100

    print(f"\n  {'─'*40}")
    print(f"  Results for: {label}")
    print(f"  {'─'*40}")
    print(f"  MAE  (Mean Absolute Error)   : {mae:>10.4f}  pts")
    print(f"  RMSE (Root Mean Sq. Error)   : {rmse:>10.4f}  pts")
    print(f"  R²   (Explained Variance)    : {r2:>10.6f}")
    print(f"  DA   (Directional Accuracy)  : {da:>9.2f} %")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "R2":  round(r2, 6),  "DA":   round(da, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
step(1, "LOAD CLEANED NIFTY 50 DATA")

df_raw = pd.read_csv(RAW_CSV, parse_dates=["Date"])
print(f"\n  Loaded raw data  : {len(df_raw):,} trading days")
print(f"  Date range       : {df_raw['Date'].min().date()}  →  {df_raw['Date'].max().date()}")
print(f"  Columns          : {list(df_raw.columns)}")
print(f"\n  Sample (last 3 rows):")
print(df_raw.tail(3).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD ENGINEERED FEATURES
# ─────────────────────────────────────────────────────────────────────────────
step(2, "LOAD TECHNICAL INDICATOR FEATURES")

"""
What are Technical Indicators?
  They are mathematical formulas applied to price & volume history
  to capture patterns that traders use. Examples in our dataset:
    • SMA / EMA   — Moving averages that smooth out price noise
    • RSI         — Momentum: is the stock overbought or oversold?
    • MACD        — Trend: is momentum accelerating or reversing?
    • ATR         — Volatility: how wild are the daily swings?
    • Bollinger Bands — Are prices unusually high/low vs recent history?
    • OBV         — Volume: is buying pressure increasing?
"""

df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
n_features = df.shape[1] - 2   # exclude Date + Target
print(f"\n  Loaded feature set : {len(df):,} rows × {df.shape[1]} columns")
print(f"  Input features     : {n_features}  technical indicators")
print(f"  Target column      : 'Target_Next_Open'  (next trading day's opening price)")
print(f"\n  Feature groups built:")
print(f"    Lag variables    : Open/High/Low/Close/Volume at t-1, t-2, t-3, t-5")
print(f"    Returns          : 1-day, 5-day, intraday % returns")
print(f"    Trend            : SMA-10, SMA-20, EMA-12, EMA-26")
print(f"    Momentum         : RSI-14, MACD line/signal/histogram, ROC-10")
print(f"    Volatility       : ATR-14, Bollinger Bands (upper/middle/lower), Rolling Std-20")
print(f"    Volume           : OBV, Volume MA-20")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD STATIONARY (SCALE-FREE) FEATURES
# ─────────────────────────────────────────────────────────────────────────────
step(3, "BUILD STATIONARY FEATURES  (normalize to Close price)")

"""
Why stationary features?
  NIFTY 50 moved from ~7,000 in 2016 to ~25,000 in 2026.
  A raw price of 8,000 looks totally different from 24,000,
  even if the market behaviour is the same. By dividing by
  the current Close price we get scale-free ratios that work
  the same regardless of when in time we are.
"""

def build_stationary_features(df):
    s = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Pass-through scale-free columns
    for col in ["Return_1d", "Return_5d", "Intraday_Return", "RSI_14", "ROC_10"]:
        s[col] = df[col]

    # Normalize OHLC & lags relative to current Close
    s["Open_to_Close"]  = (df["Open"]  - close) / close
    s["High_to_Close"]  = (df["High"]  - close) / close
    s["Low_to_Close"]   = (df["Low"]   - close) / close
    for lag in [1, 2, 3, 5]:
        s[f"Close_lag_{lag}_ratio"] = (df[f"Close_lag_{lag}"] - close) / close
        s[f"Open_lag_{lag}_ratio"]  = (df[f"Open_lag_{lag}"]  - close) / close

    # Normalize trend & volatility indicators
    s["SMA_10_ratio"]        = (df["SMA_10"]  - close) / close
    s["SMA_20_ratio"]        = (df["SMA_20"]  - close) / close
    s["EMA_12_ratio"]        = (df["EMA_12"]  - close) / close
    s["EMA_26_ratio"]        = (df["EMA_26"]  - close) / close
    s["MACD_Line_pct"]       = df["MACD_Line"]   / close
    s["MACD_Signal_pct"]     = df["MACD_Signal"] / close
    s["MACD_Hist_pct"]       = df["MACD_Hist"]   / close
    s["ATR_14_pct"]          = df["ATR_14"]       / close
    s["BB_Upper_ratio"]      = (df["BB_Upper"] - close) / close
    s["BB_Lower_ratio"]      = (df["BB_Lower"] - close) / close
    s["Rolling_Std_20_pct"]  = df["Rolling_Std_20"] / close

    # Volume ratio
    vol_ma = df["Volume_MA_20"].replace(0, np.nan)
    s["Volume_ratio"] = (df["Volume"] / vol_ma).fillna(1.0)

    # Target: overnight gap return (Close today → Open tomorrow)
    target_return = (df["Target_Next_Open"] - close) / close
    return s, target_return

X, y_ret = build_stationary_features(df)
print(f"\n  Stationary feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
print(f"  Target (gap return)       : mean = {y_ret.mean()*100:.4f}%,  std = {y_ret.std()*100:.4f}%")
print(f"\n  All {X.shape[1]} stationary features:")
for i, col in enumerate(X.columns, 1):
    print(f"    {i:2d}. {col}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — CHRONOLOGICAL TRAIN / VAL / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
step(4, "CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")

"""
Why chronological (not random)?
  Stock data is time-ordered. If we randomly shuffle rows, the
  model can "peek" at future data during training — that's cheating.
  We always train on the past and test on the future.

  Split: 70% Train | 15% Validation | 15% Test
"""

dates           = pd.to_datetime(df["Date"])
actual_next_open = df["Target_Next_Open"]
current_close    = df["Close"]

n         = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

X_train, y_train = X.iloc[:train_end],        y_ret.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y_ret.iloc[train_end:val_end]
X_test           = X.iloc[val_end:]

close_val,  open_val_true  = current_close.iloc[train_end:val_end], actual_next_open.iloc[train_end:val_end]
close_test, open_test_true = current_close.iloc[val_end:],          actual_next_open.iloc[val_end:]
test_dates                 = dates.iloc[val_end:]

print(f"\n  Total samples  : {n:,}")
print(f"  TRAIN  (70%)   : {train_end:,} days   {dates.iloc[0].date()} → {dates.iloc[train_end-1].date()}")
print(f"  VAL    (15%)   : {val_end - train_end:,} days   {dates.iloc[train_end].date()} → {dates.iloc[val_end-1].date()}")
print(f"  TEST   (15%)   : {n - val_end:,} days   {dates.iloc[val_end].date()} → {dates.iloc[-1].date()}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BASELINE RANDOM FOREST (ALL FEATURES)
# ─────────────────────────────────────────────────────────────────────────────
step(5, "BASELINE — RANDOM FOREST WITH ALL FEATURES")

"""
What is Random Forest?
  Imagine asking 200 different analysts to predict tomorrow's market.
  Each analyst looks at a random subset of your data and indicators.
  The final prediction = average of all 200 analysts. This "wisdom of
  crowds" approach is very robust and hard to overfit.

  Here we train it on ALL 28 features as a baseline to beat.
"""

print(f"\n  Training Random Forest on ALL {X.shape[1]} features ...")
t0 = time.time()
baseline_rf = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)
baseline_rf.fit(X_train, y_train)
print(f"  Done in {time.time()-t0:.1f}s")

baseline_pred_open = close_test * (1 + baseline_rf.predict(X_test))
baseline_metrics   = evaluate(open_test_true, baseline_pred_open,
                               label=f"Baseline RF — All {X.shape[1]} Features")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — RFE FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
step(6, "RFE — RECURSIVE FEATURE ELIMINATION")

"""
What is RFE?
  RFE works like pruning a tree:
    1. Train a model on ALL features
    2. Rank features by importance
    3. Remove the LEAST important feature
    4. Repeat until only k features remain

  We try k = 5, 8, 12, 16 and pick the k with the lowest
  validation error — this avoids choosing k based on test data.
"""

print(f"\n  Trying k ∈ {{5, 8, 12, 16}} features ...")
print(f"  (Using Validation RMSE to pick the best k)\n")

best_k, best_val_rmse, best_features = None, float("inf"), None
rfe_val_scores = {}

for k in [5, 8, 12, 16]:
    t0 = time.time()
    estimator = RandomForestRegressor(
        n_estimators=100, max_depth=8, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
    rfe.fit(X_train, y_train)
    selected = list(X_train.columns[rfe.support_])

    rf_tmp = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    rf_tmp.fit(X_train[selected], y_train)
    val_pred_open = close_val * (1 + rf_tmp.predict(X_val[selected]))
    val_rmse      = np.sqrt(mean_squared_error(open_val_true, val_pred_open))
    rfe_val_scores[k] = val_rmse

    marker = "  ◄── BEST" if val_rmse < best_val_rmse else ""
    print(f"  k={k:2d}  |  Validation RMSE: {val_rmse:>8.2f} pts  |  {time.time()-t0:.1f}s{marker}")

    if val_rmse < best_val_rmse:
        best_val_rmse  = val_rmse
        best_k         = k
        best_features  = selected

print(f"\n  ✓ Best k = {best_k}  (Validation RMSE = {best_val_rmse:.2f} pts)")
print(f"\n  Selected Features ({best_k}):")
for i, f in enumerate(best_features, 1):
    print(f"    {i:2d}. {f}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — FINAL RFE-RF MODEL ON TEST SET
# ─────────────────────────────────────────────────────────────────────────────
step(7, "FINAL RFE-RF MODEL — TEST SET EVALUATION")

print(f"\n  Re-training Random Forest on selected {best_k} features ...")
t0 = time.time()
rfe_rf = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_leaf=10,
    random_state=42, n_jobs=-1
)
rfe_rf.fit(X_train[best_features], y_train)
print(f"  Done in {time.time()-t0:.1f}s")

rfe_pred_open = close_test * (1 + rfe_rf.predict(X_test[best_features]))
rfe_metrics   = evaluate(open_test_true, rfe_pred_open,
                          label=f"RFE-RF — Top {best_k} Features")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────
step(8, "RESULTS COMPARISON")

results_df = pd.DataFrame([
    {"Model": f"Baseline RF  (All {X.shape[1]} features)", **baseline_metrics},
    {"Model": f"RFE-RF       (Top {best_k} features)",     **rfe_metrics},
])

print("\n")
print(results_df.to_string(index=False))
print()

# Improvement summary
mae_delta  = baseline_metrics["MAE"]  - rfe_metrics["MAE"]
rmse_delta = baseline_metrics["RMSE"] - rfe_metrics["RMSE"]
feat_reduction = round((1 - best_k / X.shape[1]) * 100, 1)

print(f"  Feature reduction : {X.shape[1]} → {best_k}  ({feat_reduction}% fewer features)")
print(f"  MAE  change       : {mae_delta:+.4f} pts")
print(f"  RMSE change       : {rmse_delta:+.4f} pts")
print(f"  R²   change       : {rfe_metrics['R2'] - baseline_metrics['R2']:+.6f}")
print(f"  DA   change       : {rfe_metrics['DA'] - baseline_metrics['DA']:+.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — SAVE RESULTS & PLOTS
# ─────────────────────────────────────────────────────────────────────────────
step(9, "SAVE RESULTS & PLOTS")

# --- metrics CSV ---
results_df.to_csv(METRICS_CSV, index=False)
print(f"\n  ✓ Metrics saved  → {METRICS_CSV}")

# --- selected features JSON ---
with open(FEATURES_JSON, "w") as f:
    json.dump({"best_k": best_k, "selected_features": best_features}, f, indent=4)
print(f"  ✓ Features saved → {FEATURES_JSON}")

# ── PLOT 1: Actual vs Predicted ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Nifty 50 — Next-Day Opening Price: Actual vs Predicted (Test Set)",
             fontsize=14, fontweight="bold", y=0.98)

for ax, (pred, label, color) in zip(
    axes,
    [(baseline_pred_open, f"Baseline RF (All {X.shape[1]} features)", "#e74c3c"),
     (rfe_pred_open,      f"RFE-RF (Top {best_k} features)",          "#2ecc71")]
):
    ax.plot(test_dates, open_test_true.values, label="Actual Next-Day Open",
            color="black", linewidth=1.8, zorder=3)
    ax.plot(test_dates, pred.values, label=label,
            color=color, linewidth=1.2, alpha=0.85, linestyle="--")
    ax.set_ylabel("Index Level (Points)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="x", rotation=30)

axes[1].set_xlabel("Date", fontsize=11)
plt.tight_layout()
plot1 = os.path.join(PLOTS_DIR, "actual_vs_predicted.png")
plt.savefig(plot1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Plot saved     → {plot1}")

# ── PLOT 2: RFE Validation Curve (k vs RMSE) ────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ks   = list(rfe_val_scores.keys())
rmses = list(rfe_val_scores.values())
bars = ax.bar(ks, rmses, color=["#2ecc71" if k == best_k else "#95a5a6" for k in ks],
              width=1.8, edgecolor="black", linewidth=0.6)
ax.set_xticks(ks)
ax.set_xticklabels([f"k={k}" for k in ks], fontsize=11)
ax.set_ylabel("Validation RMSE (pts)", fontsize=11)
ax.set_title("RFE: Validation RMSE vs Number of Selected Features", fontsize=12, fontweight="bold")
for bar, rmse in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{rmse:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(min(rmses) * 0.97, max(rmses) * 1.03)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plot2 = os.path.join(PLOTS_DIR, "rfe_validation_curve.png")
plt.savefig(plot2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Plot saved     → {plot2}")

# ── PLOT 3: Feature Importance of RFE-RF model ──────────────────────────────
importances = rfe_rf.feature_importances_
feat_imp_df = pd.DataFrame({"Feature": best_features, "Importance": importances})
feat_imp_df = feat_imp_df.sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#3498db"] * len(feat_imp_df)
ax.barh(feat_imp_df["Feature"], feat_imp_df["Importance"],
        color=colors, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=10)
ax.set_title(f"RFE-RF: Feature Importance (Top {best_k} Selected Features)", fontsize=12, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plot3 = os.path.join(PLOTS_DIR, "feature_importance.png")
plt.savefig(plot3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Plot saved     → {plot3}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\n  All outputs saved to: {RESULTS_DIR}")
print(f"\n  KEY RESULTS (Test Set):")
print(f"    Model            : RFE-RF with {best_k} features")
print(f"    MAE              : {rfe_metrics['MAE']:.2f}  pts")
print(f"    RMSE             : {rfe_metrics['RMSE']:.2f}  pts")
print(f"    R²               : {rfe_metrics['R2']:.4f}")
print(f"    Dir. Accuracy    : {rfe_metrics['DA']:.2f} %")
print()
