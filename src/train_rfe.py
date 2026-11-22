import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4), "R2": round(r2, 6)}

def build_stationary_features(df: pd.DataFrame):
    """Convert raw price-level columns into scale-invariant ratios & indicators so RF generalizes across regimes."""
    s_df = pd.DataFrame(index=df.index)
    close = df["Close"]

    # Scale-free returns & momentum/oscillators
    for col in ["Return_1d", "Return_5d", "Intraday_Return", "RSI_14", "ROC_10"]:
        s_df[col] = df[col]

    # Normalized candlestick & lag ratios relative to current Close
    s_df["Open_to_Close"] = (df["Open"] - close) / close
    s_df["High_to_Close"] = (df["High"] - close) / close
    s_df["Low_to_Close"] = (df["Low"] - close) / close
    for lag in [1, 2, 3, 5]:
        s_df[f"Close_lag_{lag}_ratio"] = (df[f"Close_lag_{lag}"] - close) / close
        s_df[f"Open_lag_{lag}_ratio"] = (df[f"Open_lag_{lag}"] - close) / close

    # Normalized Trend & Volatility indicators (relative to Close)
    s_df["SMA_10_ratio"] = (df["SMA_10"] - close) / close
    s_df["SMA_20_ratio"] = (df["SMA_20"] - close) / close
    s_df["EMA_12_ratio"] = (df["EMA_12"] - close) / close
    s_df["EMA_26_ratio"] = (df["EMA_26"] - close) / close
    s_df["MACD_Line_pct"] = df["MACD_Line"] / close
    s_df["MACD_Signal_pct"] = df["MACD_Signal"] / close
    s_df["MACD_Hist_pct"] = df["MACD_Hist"] / close
    s_df["ATR_14_pct"] = df["ATR_14"] / close
    s_df["BB_Upper_ratio"] = (df["BB_Upper"] - close) / close
    s_df["BB_Lower_ratio"] = (df["BB_Lower"] - close) / close
    s_df["Rolling_Std_20_pct"] = df["Rolling_Std_20"] / close

    # Volume ratio (handle zero-volume days safely)
    vol_ma = df["Volume_MA_20"].replace(0, np.nan)
    s_df["Volume_ratio"] = (df["Volume"] / vol_ma).fillna(1.0)

    # Stationary Target: Overnight gap return from Close(t) to Open(t+1)
    target_return = (df["Target_Next_Open"] - close) / close
    return s_df, target_return

def run_rfe_pipeline():
    os.makedirs("results/plots", exist_ok=True)
    df = pd.read_csv("data/processed/NIFTY50_features.csv")
    
    dates = pd.to_datetime(df["Date"])
    actual_next_open = df["Target_Next_Open"]
    current_close = df["Close"]

    X, y_ret = build_stationary_features(df)

    # 1. Chronological Split: 70% Train | 15% Validation | 15% Test
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y_ret.iloc[:train_end]
    X_val, y_val     = X.iloc[train_end:val_end], y_ret.iloc[train_end:val_end]
    X_test           = X.iloc[val_end:]

    close_val, open_val_true   = current_close.iloc[train_end:val_end], actual_next_open.iloc[train_end:val_end]
    close_test, open_test_true = current_close.iloc[val_end:], actual_next_open.iloc[val_end:]
    test_dates                 = dates.iloc[val_end:]

    # 2. Baseline Model: Random Forest on ALL stationary features
    print(f"Training Baseline Random Forest (All {X.shape[1]} Stationary Features)...")
    baseline_rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    baseline_rf.fit(X_train, y_train)
    
    baseline_pred_open = close_test * (1 + baseline_rf.predict(X_test))
    baseline_metrics = evaluate_predictions(open_test_true, baseline_pred_open)

    # 3. RFE Feature Selection tuned on Validation RMSE (in actual index points)
    print("Running RFE across candidate feature counts (k = 5, 8, 12, 16)...")
    best_k = None
    best_val_rmse = float("inf")
    best_rfe_features = None

    for k in [5, 8, 12, 16]:
        estimator = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=estimator, n_features_to_select=k, step=1)
        rfe.fit(X_train, y_train)
        
        selected = list(X_train.columns[rfe.support_])
        
        rf_temp = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf_temp.fit(X_train[selected], y_train)
        val_pred_open = close_val * (1 + rf_temp.predict(X_val[selected]))
        val_rmse = np.sqrt(mean_squared_error(open_val_true, val_pred_open))
        print(f"  k={k:2d} -> Validation RMSE: {val_rmse:.4f}")
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_k = k
            best_rfe_features = selected

    print(f"\nBest RFE Subset Size: k={best_k} features")
    print("Selected Features:", best_rfe_features)

    # 4. Final RFE-Random Forest Evaluation on Test Set
    rfe_rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rfe_rf.fit(X_train[best_rfe_features], y_train)
    rfe_pred_open = close_test * (1 + rfe_rf.predict(X_test[best_rfe_features]))
    rfe_metrics = evaluate_predictions(open_test_true, rfe_pred_open)

    # 5. Save Comparison Table & Selected Features
    results_df = pd.DataFrame([
        {"Model": f"Baseline RF (All {X.shape[1]} Features)", **baseline_metrics},
        {"Model": f"RFE-RF (Top {best_k} Features)", **rfe_metrics}
    ])
    print("\n--- FINAL TEST SET RESULTS (NEXT-DAY OPENING PRICE) ---")
    print(results_df.to_string(index=False))

    results_df.to_csv("results/metrics_summary.csv", index=False)
    with open("results/selected_features.json", "w") as f:
        json.dump({"best_k": best_k, "selected_features": best_rfe_features}, f, indent=4)

    # 6. Plot Actual vs. Predicted Opening Price
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, open_test_true.values, label="Actual Next-Day Open", color="black", linewidth=1.5)
    plt.plot(test_dates, baseline_pred_open.values, label="Baseline RF Predicted", linestyle="--", alpha=0.8)
    plt.plot(test_dates, rfe_pred_open.values, label=f"RFE-RF Predicted (k={best_k})", alpha=0.85)
    plt.title("Nifty 50 Next-Day Opening Price: Actual vs. Predicted (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Index Level (Points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/plots/actual_vs_predicted.png", dpi=300)
    print("\nSaved updated metrics and plot to results/")

if __name__ == "__main__":
    run_rfe_pipeline()