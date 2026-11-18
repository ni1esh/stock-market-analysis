import os
import pandas as pd
import yfinance as yf

TICKER = "^NSEI"
START_DATE = "2016-01-01"
END_DATE = "2026-11-17"

def fetch_nifty50():
    os.makedirs("data/raw", exist_ok=True)
    print(f"Downloading Nifty 50 ({TICKER}) from {START_DATE} to {END_DATE}...")
    
    df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    
    # Flatten multi-index columns if yfinance returns them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.reset_index()
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    
    # Sort chronologically and drop duplicate dates
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    
    # Remove any invalid zero/negative price rows
    price_cols = ["Open", "High", "Low", "Close"]
    df = df[(df[price_cols] > 0).all(axis=1)]
    
    # Forward-fill stray missing prices; fill missing index volume with 0
    df[price_cols] = df[price_cols].ffill()
    df["Volume"] = df["Volume"].fillna(0)
    df = df.dropna().reset_index(drop=True)
    
    out_path = "data/raw/NIFTY50_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"Done! Saved {len(df)} trading days to {out_path}")
    print(df.head(3))

if __name__ == "__main__":
    fetch_nifty50()