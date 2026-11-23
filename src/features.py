import os
import pandas as pd
import numpy as np

from ta.trend import (
    SMAIndicator, EMAIndicator, MACD, ADXIndicator, CCIIndicator,
    AroonIndicator, PSARIndicator
)
from ta.momentum import (
    RSIIndicator, ROCIndicator, StochasticOscillator, WilliamsRIndicator, TSIIndicator
)
from ta.volatility import (
    AverageTrueRange, BollingerBands, KeltnerChannel, DonchianChannel
)
from ta.volume import (
    OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, MFIIndicator,
    VolumePriceTrendIndicator
)

def generate_nifty_features():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "src", "data", "raw", "NIFTY50_clean.csv")
    out_dir = os.path.join(base_dir, "src", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "NIFTY50_features.csv")

    if not os.path.exists(raw_path):
        raw_path = os.path.join("data", "raw", "NIFTY50_clean.csv")
        out_path = os.path.join("data", "processed", "NIFTY50_features.csv")

    df = pd.read_csv(raw_path)
    
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]
    volume = df["Volume"]

    # =========================================================================
    # 1. Lagged Market Variables (t-1, t-2, t-3, t-5)
    # =========================================================================
    for lag in [1, 2, 3, 5]:
        df[f"Open_lag_{lag}"] = open_p.shift(lag)
        df[f"High_lag_{lag}"] = high.shift(lag)
        df[f"Low_lag_{lag}"] = low.shift(lag)
        df[f"Close_lag_{lag}"] = close.shift(lag)
        df[f"Volume_lag_{lag}"] = volume.shift(lag)

    # =========================================================================
    # 2. Historical Returns
    # =========================================================================
    df["Return_1d"] = ((close - close.shift(1)) / close.shift(1)) * 100
    df["Return_3d"] = ((close - close.shift(3)) / close.shift(3)) * 100
    df["Return_5d"] = ((close - close.shift(5)) / close.shift(5)) * 100
    df["Return_10d"] = ((close - close.shift(10)) / close.shift(10)) * 100
    df["Intraday_Return"] = ((close - open_p) / open_p) * 100

    # =========================================================================
    # 3. Trend Indicators (Moving Averages, MACD, ADX, Aroon, PSAR, CCI)
    # =========================================================================
    # Simple Moving Averages
    df["SMA_5"] = SMAIndicator(close=close, window=5).sma_indicator()
    df["SMA_10"] = SMAIndicator(close=close, window=10).sma_indicator()
    df["SMA_20"] = SMAIndicator(close=close, window=20).sma_indicator()
    df["SMA_50"] = SMAIndicator(close=close, window=50).sma_indicator()
    
    # Exponential Moving Averages
    df["EMA_5"] = EMAIndicator(close=close, window=5).ema_indicator()
    df["EMA_12"] = EMAIndicator(close=close, window=12).ema_indicator()
    df["EMA_26"] = EMAIndicator(close=close, window=26).ema_indicator()
    df["EMA_50"] = EMAIndicator(close=close, window=50).ema_indicator()

    # MACD
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD_Line"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # ADX (Trend Strength)
    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX_14"] = adx.adx()
    df["ADX_Pos_DI"] = adx.adx_pos()
    df["ADX_Neg_DI"] = adx.adx_neg()

    # CCI
    df["CCI_20"] = CCIIndicator(high=high, low=low, close=close, window=20).cci()

    # Aroon Indicator (uses high and low)
    aroon = AroonIndicator(high=high, low=low, window=25)
    df["Aroon_Up"] = aroon.aroon_up()
    df["Aroon_Down"] = aroon.aroon_down()
    df["Aroon_Osc"] = df["Aroon_Up"] - df["Aroon_Down"]

    # Parabolic SAR
    psar = PSARIndicator(high=high, low=low, close=close)
    df["PSAR"] = psar.psar()

    # =========================================================================
    # 4. Momentum Oscillators (RSI, Stochastic, Williams %R, ROC, TSI)
    # =========================================================================
    df["RSI_7"] = RSIIndicator(close=close, window=7).rsi()
    df["RSI_14"] = RSIIndicator(close=close, window=14).rsi()
    df["ROC_5"] = ROCIndicator(close=close, window=5).roc()
    df["ROC_10"] = ROCIndicator(close=close, window=10).roc()

    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    df["Williams_R_14"] = WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r()
    df["TSI_25_13"] = TSIIndicator(close=close, window_slow=25, window_fast=13).tsi()

    # =========================================================================
    # 5. Volatility Indicators (ATR, Bollinger Bands, Keltner, Donchian, Rolling Std)
    # =========================================================================
    df["ATR_14"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Width"] = bb.bollinger_wband()
    df["BB_Pct"] = bb.bollinger_pband()

    kc = KeltnerChannel(high=high, low=low, close=close, window=20)
    df["KC_Middle"] = kc.keltner_channel_mband()
    df["KC_Upper"] = kc.keltner_channel_hband()
    df["KC_Lower"] = kc.keltner_channel_lband()

    dc = DonchianChannel(high=high, low=low, close=close, window=20)
    df["DC_Upper"] = dc.donchian_channel_hband()
    df["DC_Lower"] = dc.donchian_channel_lband()
    df["DC_Middle"] = dc.donchian_channel_mband()

    df["Rolling_Std_10"] = close.rolling(window=10).std()
    df["Rolling_Std_20"] = close.rolling(window=20).std()

    # =========================================================================
    # 6. Volume Indicators (OBV, Volume MA, CMF, MFI, VPT)
    # =========================================================================
    df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df["Volume_MA_10"] = volume.rolling(window=10).mean()
    df["Volume_MA_20"] = volume.rolling(window=20).mean()
    df["CMF_20"] = ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume, window=20).chaikin_money_flow()
    df["MFI_14"] = MFIIndicator(high=high, low=low, close=close, volume=volume, window=14).money_flow_index()
    df["VPT"] = VolumePriceTrendIndicator(close=close, volume=volume).volume_price_trend()

    # =========================================================================
    # 7. Target Variable: Next Trading Day's Open Price
    # =========================================================================
    df["Target_Next_Open"] = open_p.shift(-1)

    # Drop warm-up rows (around 50 for SMA_50) and last row
    df = df.dropna().reset_index(drop=True)

    df.to_csv(out_path, index=False)
    total_features = df.shape[1] - 2  # minus Date and Target
    print(f"Done! Saved {len(df)} rows with {total_features} total features to {out_path}")

if __name__ == "__main__":
    generate_nifty_features()
