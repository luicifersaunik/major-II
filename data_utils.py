"""
Data Pipeline — Stock Closing Price Prediction
Handles normalization, sliding window generation, and synthetic data fallback.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
#  Normalisation (Eq. 24)
# ─────────────────────────────────────────────
def normalize(x: np.ndarray, y_min: float = -1.0, y_max: float = 1.0):
    """Min-max scale x to [y_min, y_max]. Returns scaled array + scale params."""
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        return np.zeros_like(x), x_min, x_max
    scaled = (y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min
    return scaled, x_min, x_max

def denormalize(y_scaled: np.ndarray, x_min: float, x_max: float,
                y_min: float = -1.0, y_max: float = 1.0):
    """Reverse min-max scaling."""
    return (y_scaled - y_min) / (y_max - y_min) * (x_max - x_min) + x_min


# ─────────────────────────────────────────────
#  Sliding Window (Fig. 3)
# ─────────────────────────────────────────────
def sliding_window(series: np.ndarray, window: int = 6):
    """
    Create (X, y) pairs using a sliding window.
    X[i] = series[i : i+window]
    y[i] = series[i+window]
    """
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append([series[i + window]])
    return np.array(X), np.array(y)


def train_test_split_sequential(X, y, test_ratio: float = 0.2):
    """Time-series safe split — no shuffling."""
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], y[:split], X[split:], y[split:]


# ─────────────────────────────────────────────
#  Synthetic stock data generator
# ─────────────────────────────────────────────
def generate_synthetic_stock(n: int = 500, name: str = "SynthStock",
                               seed: int = 42) -> pd.Series:
    """
    Geometric Brownian Motion (GBM) — realistic synthetic price series.
    mu=annual drift, sigma=annual volatility, dt=1/252 trading days.
    """
    rng = np.random.default_rng(seed)
    mu, sigma, dt = 0.08, 0.20, 1/252
    S = np.zeros(n)
    S[0] = 100.0
    for t in range(1, n):
        z = rng.standard_normal()
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return pd.Series(S, name=name)


# ─────────────────────────────────────────────
#  Dataset catalogue (paper Table 1 metadata)
# ─────────────────────────────────────────────
PAPER_DATASETS = {
    "BSE":         {"full": "Bombay Stock Exchange",                   "n": 2454, "max": 41953,  "min": 15175},
    "NASDAQ":      {"full": "NASDAQ Composite",                        "n": 2517, "max": 9817.2, "min": 2091.8},
    "HSI":         {"full": "Hang Seng Index",                         "n": 2459, "max": 33154,  "min": 16250},
    "SSE":         {"full": "Shanghai Composite Index",                "n": 2859, "max": 5166.35,"min": 1950.012},
    "Russell2000": {"full": "Russell 2000 Index",                      "n": 2965, "max": 2442.74,"min": 590.03},
    "TAIEX":       {"full": "Taiwan Capitalization Weighted Stock Index","n": 1068,"max": 27.5,   "min": 16.66},
}


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────
def nmse(y_true, y_pred):
    """Normalised Mean Square Error (paper's performance metric)."""
    return float(np.mean((y_true - y_pred) ** 2) / (np.var(y_true) + 1e-12))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true, y_pred):
    mask = np.abs(y_true) > 1e-6
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
