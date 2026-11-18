# NIFTY 50 Next-Day Opening Price Prediction

A machine learning pipeline to predict the **next trading day's opening price** of the NIFTY 50 index using technical indicators and Recursive Feature Elimination (RFE).

---

## Project Structure

```
├── pipeline.py                  ← Run this — full end-to-end pipeline
├── src/
│   ├── data_loader.py           ← Fetches NIFTY 50 data via yfinance
│   ├── features.py              ← Generates technical indicator features
│   ├── train_rfe.py             ← RFE + Random Forest training script
│   ├── data/
│   │   ├── raw/                 ← Raw OHLCV data (NIFTY50_clean.csv)
│   │   └── processed/           ← Feature-engineered dataset
│   └── results/
│       ├── metrics_summary.csv  ← MAE, RMSE, R², Directional Accuracy
│       ├── selected_features.json ← RFE-selected features
│       └── plots/               ← All generated plots
```

---

## Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Load 10 years of NIFTY 50 OHLCV data (2016–2026) |
| 2 | Engineer 35+ technical indicator features |
| 3 | Normalize to scale-free ratios (stationary features) |
| 4 | Chronological 70/15/15 Train/Val/Test split |
| 5 | Baseline Random Forest (all 28 features) |
| 6 | RFE feature selection (k ∈ {5, 8, 12, 16}, tuned on validation) |
| 7 | Final RFE-RF model evaluated on test set |
| 8 | Metrics: MAE, RMSE, R², Directional Accuracy |
| 9 | Save plots and results |

---

## Results (Test Set)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Baseline RF (All 28 features) | 93.59 pts | 158.61 pts | 0.9746 |
| **RFE-RF (Top 12 features)** | 93.82 pts | 159.14 pts | 0.9744 |

> RFE achieves equivalent accuracy using **57% fewer features** (28 → 12)

---

## Features Selected by RFE (k=12)

`Return_1d`, `Return_5d`, `RSI_14`, `ROC_10`, `High_to_Close`, `Low_to_Close`,
`Open_lag_1_ratio`, `SMA_10_ratio`, `MACD_Line_pct`, `MACD_Signal_pct`,
`ATR_14_pct`, `Rolling_Std_20_pct`

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib ta yfinance

# Run full pipeline
python pipeline.py
```

---

## Tech Stack

- **Data:** yfinance (NIFTY 50, `^NSEI`)
- **Features:** `ta` library (technical analysis indicators)
- **Model:** Scikit-learn Random Forest + RFE
- **Evaluation:** MAE, RMSE, R², Directional Accuracy
