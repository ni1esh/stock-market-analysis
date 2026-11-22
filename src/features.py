import os
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

def generate_nifty_features():
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/raw/NIFTY50_clean.csv")
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]
    volume = df["Volume"]

    # 1. Lagged Market Variables (Section 3.7.1: t-1, t-2, t-3, t-5)
    for lag in [1, 2, 3, 5]:
        df[f"Open_lag_{lag}"] = open_p.shift(lag)
        df[f"High_lag_{lag}"] = high.shift(lag)
        df[f"Low_lag_{lag}"] = low.shift(lag)
        df[f"Close_lag_{lag}"] = close.shift(lag)
        df[f"Volume_lag_{lag}"] = volume.shift(lag)

    # 2. Historical Returns (Section 3.7.2)
    df["Return_1d"] = ((close - close.shift(1)) / close.shift(1)) * 100
    df["Return_5d"] = ((close - close.shift(5)) / close.shift(5)) * 100
    df["Intraday_Return"] = ((close - open_p) / open_p) * 100

    # 3. Trend Indicators (Appendix A.2 SMA, A.3 EMA)
    df["SMA_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["EMA_12"] = EMAIndicator(close=close, window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close=close, window=26).ema_indicator()

    # 4. Momentum Indicators (Appendix A.4 RSI, A.5 MACD, A.6 ROC)
    df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD_Line"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    df["ROC_10"] = ROCIndicator(close=close, window=10).roc()

    # 5. Volatility Indicators (Appendix A.7 ATR, A.8 Bollinger Bands, A.9 Rolling Std)
    df["ATR_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["Rolling_Std_20"] = close.rolling(window=20).std()

    # 6. Volume Indicators (Appendix A.10 OBV, A.11 Volume MA)
    df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["Volume_MA_20"] = volume.rolling(window=20).mean()

    # 7. Target Variable: Next Trading Day's Open Price
    df["Target_Next_Open"] = open_p.shift(-1)

    # Drop the first ~33 warm-up rows (from 26-period MACD + 9-period signal) and last row
    df = df.dropna().reset_index(drop=True)
    
    out_path = "data/processed/NIFTY50_features.csv"
    df.to_csv(out_path, index=False)
    print(f"Done! {len(df)} rows and {df.shape[1] - 2} input features saved to {out_path}")

if __name__ == "__main__":
    generate_nifty_features()