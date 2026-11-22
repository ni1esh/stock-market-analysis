"""
Boruta Feature Selection Pipeline
---------------------------------
This script implements the Boruta algorithm for feature selection on the NIFTY 50 dataset.

How Boruta Works:
1. Creates 'shadow features' by shuffling values of each real feature.
2. Trains an estimator (tree ensemble) on real + shadow features.
3. Compares the importance of real features against the maximum importance of shadow features.
4. Statistically accepts confirmed features and rejects noise features across multiple iterations.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────────────────────────────────────
# Boruta Feature Selection Algorithm (Pure Python + scikit-learn)
# No external 'boruta' package needed!
# ─────────────────────────────────────────────────────────────────────────────

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_CSV = os.path.join(BASE_DIR, "data", "processed", "NIFTY50_features.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    actual_dir = np.sign(np.array(y_true).flatten()[1:] - np.array(y_true).flatten()[:-1])
    predicted_dir = np.sign(np.array(y_pred).flatten()[1:] - np.array(y_pred).flatten()[:-1])
    da = np.mean(actual_dir == predicted_dir) * 100

    print(f"\n  --- Results: {label} ---")
    print(f"  MAE  : {mae:>10.4f} pts")
    print(f"  RMSE : {rmse:>10.4f} pts")
    print(f"  R²   : {r2:>10.6f}")
    print(f"  DA   : {da:>9.2f} %")
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 6), "DA": round(da, 2)}


def build_stationary_features(df):
    """Convert raw prices to scale-invariant ratios for stable modeling."""
    s = pd.DataFrame(index=df.index)
    close = df["Close"]

    for col in ["Return_1d", "Return_5d", "Intraday_Return", "RSI_14", "ROC_10"]:
        s[col] = df[col]

    s["Open_to_Close"] = (df["Open"] - close) / close
    s["High_to_Close"] = (df["High"] - close) / close
    s["Low_to_Close"] = (df["Low"] - close) / close
    for lag in [1, 2, 3, 5]:
        s[f"Close_lag_{lag}_ratio"] = (df[f"Close_lag_{lag}"] - close) / close
        s[f"Open_lag_{lag}_ratio"] = (df[f"Open_lag_{lag}"] - close) / close

    s["SMA_10_ratio"] = (df["SMA_10"] - close) / close
    s["SMA_20_ratio"] = (df["SMA_20"] - close) / close
    s["EMA_12_ratio"] = (df["EMA_12"] - close) / close
    s["EMA_26_ratio"] = (df["EMA_26"] - close) / close
    s["MACD_Line_pct"] = df["MACD_Line"] / close
    s["MACD_Signal_pct"] = df["MACD_Signal"] / close
    s["MACD_Hist_pct"] = df["MACD_Hist"] / close
    s["ATR_14_pct"] = df["ATR_14"] / close
    s["BB_Upper_ratio"] = (df["BB_Upper"] - close) / close
    s["BB_Lower_ratio"] = (df["BB_Lower"] - close) / close
    s["Rolling_Std_20_pct"] = df["Rolling_Std_20"] / close

    vol_ma = df["Volume_MA_20"].replace(0, np.nan)
    s["Volume_ratio"] = (df["Volume"] / vol_ma).fillna(1.0)

    target_return = (df["Target_Next_Open"] - close) / close
    return s, target_return


def run_custom_boruta(X_train, y_train, max_iter=50, random_state=42):
    """
    Pure Python/scikit-learn Boruta implementation in case boruta-py package isn't installed.
    """
    print("  Using built-in Boruta shadow feature algorithm...")
    rng = np.random.RandomState(random_state)
    n_features = X_train.shape[1]
    feature_names = list(X_train.columns)
    
    hits = np.zeros(n_features)
    
    for it in range(max_iter):
        # Create shadow features by randomly permuting columns
        X_shadow = np.array(X_train.values, copy=True)
        for col in range(X_shadow.shape[1]):
            perm = rng.permutation(X_shadow.shape[0])
            X_shadow[:, col] = X_shadow[perm, col]
            
        # Combine real + shadow
        X_combined = np.hstack([X_train.values, X_shadow])
        
        # Train estimator
        rf = RandomForestRegressor(n_estimators=100, max_depth=7, min_samples_leaf=10, random_state=random_state + it, n_jobs=-1)
        rf.fit(X_combined, y_train)
        
        # Feature importances
        importances = rf.feature_importances_
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]
        
        # Max shadow importance is the noise threshold
        max_shadow_imp = np.max(shadow_imp)
        
        # Count hits where real feature outperformed noise threshold
        hits += (real_imp > max_shadow_imp).astype(int)
        
        if (it + 1) % 10 == 0 or it == max_iter - 1:
            print(f"    Iteration {it+1}/{max_iter} complete")

    # Select features that beat shadow threshold in significant number of runs (> 50%)
    threshold_hits = max_iter * 0.5
    selected_mask = hits >= threshold_hits
    
    # If too few or none selected due to strictness, pick at least top features with most hits
    if np.sum(selected_mask) < 3:
        top_idx = np.argsort(hits)[-8:]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[top_idx] = True

    selected_features = [feature_names[i] for i, sel in enumerate(selected_mask) if sel]
    ranking = max_iter - hits + 1
    return selected_features, ranking, hits


def run_boruta_pipeline():
    print("=" * 60)
    print("  BORUTA FEATURE SELECTION PIPELINE")
    print("=" * 60)

    # 1. Load Data
    df = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    dates = pd.to_datetime(df["Date"])
    actual_next_open = df["Target_Next_Open"]
    current_close = df["Close"]

    X, y_ret = build_stationary_features(df)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y_ret.iloc[:train_end]
    X_test = X.iloc[val_end:]
    close_test = current_close.iloc[val_end:]
    open_test_true = actual_next_open.iloc[val_end:]
    test_dates = dates.iloc[val_end:]

    print(f"Total features: {X.shape[1]}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # 2. Run Boruta
    t0 = time.time()
    selected_features, ranking, hits = run_custom_boruta(X_train, y_train, max_iter=40, random_state=42)

    elapsed = time.time() - t0
    print(f"\n  Boruta selection completed in {elapsed:.1f}s")
    print(f"  Selected {len(selected_features)} features:")
    for i, col in enumerate(selected_features, 1):
        print(f"    {i:2d}. {col}")

    # 3. Train Baseline (All Features) vs Boruta Selected
    print("\n  Training Baseline Model (All Features)...")
    baseline_model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    baseline_model.fit(X_train, y_train)
    baseline_preds = close_test * (1 + baseline_model.predict(X_test))
    baseline_metrics = evaluate(open_test_true, baseline_preds, label="Baseline (All Features)")

    print(f"\n  Training Boruta Model ({len(selected_features)} Selected Features)...")
    boruta_model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    boruta_model.fit(X_train[selected_features], y_train)
    boruta_preds = close_test * (1 + boruta_model.predict(X_test[selected_features]))
    boruta_metrics = evaluate(open_test_true, boruta_preds, label=f"Boruta ({len(selected_features)} Features)")

    # 4. Save results
    results_df = pd.DataFrame([
        {"Method": f"Baseline (All {X.shape[1]} Features)", **baseline_metrics},
        {"Method": f"Boruta (Top {len(selected_features)} Features)", **boruta_metrics}
    ])
    
    out_csv = os.path.join(RESULTS_DIR, "boruta_metrics_summary.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Summary saved to {out_csv}")
    print(results_df.to_string(index=False))

    out_json = os.path.join(RESULTS_DIR, "boruta_selected_features.json")
    with open(out_json, "w") as f:
        json.dump({"method": "Boruta", "k": len(selected_features), "selected_features": selected_features}, f, indent=4)
    print(f"  Selected features saved to {out_json}")

    # 5. Plot Actual vs Boruta Predicted
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, open_test_true.values, label="Actual Next-Day Open", color="black", linewidth=1.5)
    plt.plot(test_dates, boruta_preds.values, label=f"Boruta Predicted ({len(selected_features)} features)", color="#27ae60", linestyle="--", alpha=0.9)
    plt.title("Nifty 50 Next-Day Open: Boruta Feature Selection", fontsize=12, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Index Level (Points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "boruta_actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved to {plot_path}")


if __name__ == "__main__":
    run_boruta_pipeline()
