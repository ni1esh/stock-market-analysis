import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    actual_dir = np.sign(np.array(y_true).flatten()[1:] - np.array(y_true).flatten()[:-1])
    predicted_dir = np.sign(np.array(y_pred).flatten()[1:] - np.array(y_pred).flatten()[:-1])
    da = np.mean(actual_dir == predicted_dir) * 100
    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "RMSE": round(rmse, 4), "R2": round(r2, 6), "DA": round(da, 2)}

def build_stationary_features(df: pd.DataFrame):
    """
    Convert raw price-level features into stationary ratios & percentages
    relative to current Close price, ensuring the model generalizes over time.
    """
    s_df = pd.DataFrame(index=df.index)
    close = df["Close"]

    # 1. Direct percentage/momentum/oscillator features (already scale-free)
    scale_free_cols = [
        "Return_1d", "Return_3d", "Return_5d", "Return_10d", "Intraday_Return",
        "RSI_7", "RSI_14", "ROC_5", "ROC_10", "Stoch_K", "Stoch_D", "Williams_R_14", "TSI_25_13",
        "ADX_14", "ADX_Pos_DI", "ADX_Neg_DI", "CCI_20", "Aroon_Up", "Aroon_Down", "Aroon_Osc",
        "BB_Width", "BB_Pct", "CMF_20", "MFI_14"
    ]
    for col in scale_free_cols:
        if col in df.columns:
            s_df[col] = df[col]

    # 2. Normalized Candlestick ratios
    s_df["Open_to_Close"] = (df["Open"] - close) / close
    s_df["High_to_Close"] = (df["High"] - close) / close
    s_df["Low_to_Close"] = (df["Low"] - close) / close

    # 3. Lagged price ratios
    for lag in [1, 2, 3, 5]:
        if f"Close_lag_{lag}" in df.columns:
            s_df[f"Close_lag_{lag}_ratio"] = (df[f"Close_lag_{lag}"] - close) / close
        if f"Open_lag_{lag}" in df.columns:
            s_df[f"Open_lag_{lag}_ratio"] = (df[f"Open_lag_{lag}"] - close) / close

    # 4. Moving Average Ratios (Trend)
    for ma_col in ["SMA_5", "SMA_10", "SMA_20", "SMA_50", "EMA_5", "EMA_12", "EMA_26", "EMA_50", "PSAR"]:
        if ma_col in df.columns:
            s_df[f"{ma_col}_ratio"] = (df[ma_col] - close) / close

    # MACD ratios
    if "MACD_Line" in df.columns:
        s_df["MACD_Line_pct"] = df["MACD_Line"] / close
        s_df["MACD_Signal_pct"] = df["MACD_Signal"] / close
        s_df["MACD_Hist_pct"] = df["MACD_Hist"] / close

    # 5. Volatility Indicators normalized
    if "ATR_14" in df.columns:
        s_df["ATR_14_pct"] = df["ATR_14"] / close
    for b_col in ["BB_Middle", "BB_Upper", "BB_Lower"]:
        if b_col in df.columns:
            s_df[f"{b_col}_ratio"] = (df[b_col] - close) / close
    for k_col in ["KC_Middle", "KC_Upper", "KC_Lower", "DC_Upper", "DC_Lower", "DC_Middle"]:
        if k_col in df.columns:
            s_df[f"{k_col}_ratio"] = (df[k_col] - close) / close
    for std_col in ["Rolling_Std_10", "Rolling_Std_20"]:
        if std_col in df.columns:
            s_df[f"{std_col}_pct"] = df[std_col] / close

    # 6. Volume normalized
    if "Volume_MA_10" in df.columns:
        vol_ma10 = df["Volume_MA_10"].replace(0, np.nan)
        s_df["Volume_ratio_10"] = (df["Volume"] / vol_ma10).fillna(1.0)
    if "Volume_MA_20" in df.columns:
        vol_ma20 = df["Volume_MA_20"].replace(0, np.nan)
        s_df["Volume_ratio_20"] = (df["Volume"] / vol_ma20).fillna(1.0)

    # Clean any inf or NaNs
    s_df = s_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Stationary Target: Overnight gap return from Close(t) to Open(t+1)
    target_return = (df["Target_Next_Open"] - close) / close
    return s_df, target_return

def run_rfe_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "src", "results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    features_csv = os.path.join(base_dir, "src", "data", "processed", "NIFTY50_features.csv")
    if not os.path.exists(features_csv):
        features_csv = os.path.join("data", "processed", "NIFTY50_features.csv")

    df = pd.read_csv(features_csv)
    dates = pd.to_datetime(df["Date"])
    actual_next_open = df["Target_Next_Open"]
    current_close = df["Close"]

    X, y_ret = build_stationary_features(df)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y_ret.iloc[:train_end]
    X_val, y_val     = X.iloc[train_end:val_end], y_ret.iloc[train_end:val_end]
    X_test           = X.iloc[val_end:]

    close_val, open_val_true   = current_close.iloc[train_end:val_end], actual_next_open.iloc[train_end:val_end]
    close_test, open_test_true = current_close.iloc[val_end:], actual_next_open.iloc[val_end:]
    test_dates                 = dates.iloc[val_end:]

    print(f"Total candidate features generated: {X.shape[1]}")
    print(f"Dataset split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 1. Baseline Model: Random Forest on ALL features
    print(f"\n[1/4] Training Baseline Random Forest (All {X.shape[1]} Features)...")
    baseline_rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    baseline_rf.fit(X_train, y_train)
    baseline_pred_open = close_test * (1 + baseline_rf.predict(X_test))
    baseline_metrics = evaluate_predictions(open_test_true, baseline_pred_open)
    print(f"Baseline Test MAE: {baseline_metrics['MAE']:.2f}, RMSE: {baseline_metrics['RMSE']:.2f}, R2: {baseline_metrics['R2']:.4f}")

    # 2. Automated RFE Grid Search across candidate subset sizes
    # Tests a comprehensive range from very sparse (5) to full retention
    candidate_ks = [5, 10, 15, 20, 25, 30]
    candidate_ks = [k for k in candidate_ks if k < X.shape[1]]
    print(f"\n[2/4] Running RFE search over candidate subset sizes: {candidate_ks}...")

    best_k = None
    best_val_rmse = float("inf")
    best_rfe_features = None
    val_scores = {}

    for k in candidate_ks:
        estimator = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
        rfe = RFE(estimator=estimator, n_features_to_select=k, step=2)
        rfe.fit(X_train, y_train)
        selected = list(X_train.columns[rfe.support_])

        rf_val = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf_val.fit(X_train[selected], y_train)
        val_pred_open = close_val * (1 + rf_val.predict(X_val[selected]))
        val_rmse = np.sqrt(mean_squared_error(open_val_true, val_pred_open))
        val_scores[k] = val_rmse
        print(f"  k = {k:2d} -> Validation RMSE: {val_rmse:.4f} pts")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_k = k
            best_rfe_features = selected

    print(f"\n--> Mathematically Optimal Feature Subset: k = {best_k} (Val RMSE: {best_val_rmse:.4f})")
    print(f"--> Selected Features ({best_k}):\n", best_rfe_features)

    # 3. Final Model on Selected Features
    print(f"\n[3/4] Evaluating Final RFE-RF Model (k={best_k}) on held-out Test Set...")
    final_rf = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=10, random_state=42, n_jobs=-1)
    final_rf.fit(X_train[best_rfe_features], y_train)
    rfe_pred_open = close_test * (1 + final_rf.predict(X_test[best_rfe_features]))
    rfe_metrics = evaluate_predictions(open_test_true, rfe_pred_open)

    # Comparison Table
    results_df = pd.DataFrame([
        {"Model": f"Baseline RF (All {X.shape[1]} Features)", **baseline_metrics},
        {"Model": f"RFE-RF (Optimal {best_k} Features)", **rfe_metrics}
    ])
    print("\n--- FINAL COMPARISON RESULTS (TEST SET) ---")
    print(results_df.to_string(index=False))

    metrics_csv = os.path.join(results_dir, "metrics_summary.csv")
    results_df.to_csv(metrics_csv, index=False)

    feat_json = os.path.join(results_dir, "selected_features.json")
    with open(feat_json, "w") as f:
        json.dump({"total_candidates": X.shape[1], "optimal_k": best_k, "selected_features": best_rfe_features}, f, indent=4)

    # 4. Generate & Save Plots
    print("\n[4/4] Updating plots...")
    
    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, open_test_true.values, label="Actual Next-Day Open", color="black", linewidth=1.5)
    plt.plot(test_dates, baseline_pred_open.values, label=f"Baseline RF ({X.shape[1]} Feats)", linestyle=":", alpha=0.7)
    plt.plot(test_dates, rfe_pred_open.values, label=f"Optimal RFE-RF ({best_k} Feats)", color="#27ae60", linestyle="--", alpha=0.9)
    plt.title("Nifty 50 Next-Day Open: Actual vs. Predicted (Test Set)", fontsize=12, fontweight="bold")
    plt.xlabel("Date")
    plt.ylabel("Index Level (Points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "actual_vs_predicted.png"), dpi=200)
    plt.close()

    # Plot 2: Validation Curve
    plt.figure(figsize=(8, 4))
    ks = list(val_scores.keys())
    scores = list(val_scores.values())
    plt.plot(ks, scores, marker='o', linewidth=2, color="#2980b9")
    plt.axvline(best_k, color="red", linestyle="--", label=f"Optimal k={best_k}")
    plt.title("RFE Validation Curve (Feature Count vs. RMSE)", fontsize=11, fontweight="bold")
    plt.xlabel("Number of Selected Features (k)")
    plt.ylabel("Validation RMSE (Points)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rfe_validation_curve.png"), dpi=200)
    plt.close()

    # Plot 3: Feature Importances
    imp = final_rf.feature_importances_
    feat_imp = pd.Series(imp, index=best_rfe_features).sort_values(ascending=True)
    plt.figure(figsize=(9, max(5, int(len(feat_imp) * 0.35))))
    feat_imp.plot(kind="barh", color="#16a085", edgecolor="black", linewidth=0.5)
    plt.title(f"Feature Importance Scores (Top {best_k} Selected by RFE)", fontsize=11, fontweight="bold")
    plt.xlabel("Relative Importance (MDI)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance.png"), dpi=200)
    plt.close()

    print("Pipeline finished successfully! All artifacts generated in src/results/")

if __name__ == "__main__":
    run_rfe_pipeline()
