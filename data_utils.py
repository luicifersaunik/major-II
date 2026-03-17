"""
data_utils.py — Data Pipeline with Real Market Data via yfinance
Supports both real data (yfinance) and synthetic GBM fallback.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Ticker map
# ─────────────────────────────────────────────
TICKERS = {
    "NIFTY": "^NSEI",
    "NASDAQ":      "^IXIC",
    "HSI":         "^HSI",
    "SSE":         "000001.SS",
    "Russell2000": "^RUT",
    "TAIEX":       "^TWII",
}


def download_real_data(stock_name, start="2015-01-01", end="2023-01-01"):
    """
    Download real closing prices from Yahoo Finance via yfinance.
    Automatically falls back to synthetic GBM if download fails.
    """
    try:
        import yfinance as yf
        ticker = TICKERS.get(stock_name)
        if ticker is None:
            raise ValueError(f"Unknown stock: {stock_name}")

        print(f"  Downloading {stock_name} ({ticker})...", end=" ", flush=True)
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)

        if df.empty:
            raise ValueError("Empty dataframe — ticker may be wrong or date range invalid")

        series = df["Close"].dropna().values.flatten()
        print(f"{len(series)} trading days  ✓")
        return series.astype(float)

    except ImportError:
        print("\n  yfinance not installed — run: pip install yfinance")
        print("  Falling back to synthetic data...")
    except Exception as e:
        print(f"\n  Download failed: {e}")
        print("  Falling back to synthetic data...")

    seeds = {"BSE":1,"NASDAQ":2,"HSI":3,"SSE":4,"Russell2000":5,"TAIEX":6}
    return generate_synthetic_stock(n=400, seed=seeds.get(stock_name, 42)).values


# ─────────────────────────────────────────────
#  Synthetic GBM fallback
# ─────────────────────────────────────────────
def generate_synthetic_stock(n=400, name="SynthStock", seed=42):
    """Geometric Brownian Motion — realistic synthetic price series."""
    rng = np.random.default_rng(seed)
    mu, sigma, dt = 0.08, 0.20, 1/252
    S = np.zeros(n)
    S[0] = 100.0
    for t in range(1, n):
        z = rng.standard_normal()
        S[t] = S[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*z)
    return pd.Series(S, name=name)


# ─────────────────────────────────────────────
#  Normalisation (Eq. 24)
# ─────────────────────────────────────────────
def normalize(x, y_min=-1.0, y_max=1.0):
    x_min, x_max = float(x.min()), float(x.max())
    if x_max == x_min:
        return np.zeros_like(x, dtype=float), x_min, x_max
    scaled = (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min
    return scaled.astype(float), x_min, x_max

def denormalize(y_scaled, x_min, x_max, y_min=-1.0, y_max=1.0):
    return (y_scaled - y_min) / (y_max - y_min) * (x_max - x_min) + x_min


# ─────────────────────────────────────────────
#  Sliding window (Fig. 3 of paper)
# ─────────────────────────────────────────────
def sliding_window(series, window=6):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append([series[i + window]])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def train_test_split_sequential(X, y, test_ratio=0.2):
    split = int(len(X) * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────
def nmse(y_true, y_pred):
    return float(np.mean((y_true - y_pred)**2) / (np.var(y_true) + 1e-12))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
