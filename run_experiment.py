"""
run_experiment_fast.py
Fast version for quick testing — runs in ~20-30 seconds.
Reduce EPOCHS and dataset size for speed.
"""

import json, time, numpy as np
from data_utils import (
    generate_synthetic_stock, normalize, denormalize,
    sliding_window, train_test_split_sequential,
    nmse, rmse, mae, mape
)
from qenn_model import QENN
from enn_baseline import ENN

# ── Speed settings (change these) ─────────────────────────
WINDOW   = 6
EPOCHS   = 30       # was 200 → now 30  (main speedup)
N_POINTS = 200      # was 600 → now 200 (smaller dataset)
SEED     = 42
NH_ENN   = [10, 20 , 40 , 50 , 70] # was [10,20,40,70,100] →   // 5 models
# ──────────────────────────────────────────────────────────

STOCKS = {
    "BSE":         1,
    "NASDAQ":      2,
    "HSI":         3,
    "SSE":         4,
    "Russell2000": 5,
    "TAIEX":       6,
}

def prepare(series_raw):
    s, xmin, xmax = normalize(series_raw)
    X, y = sliding_window(s, WINDOW)
    Xtr, ytr, Xte, yte = train_test_split_sequential(X, y)
    return Xtr, ytr, Xte, yte, xmin, xmax

def evaluate(model, Xte, yte, xmin, xmax, history):
    p  = model.predict(Xte).flatten()
    yf = yte.flatten()
    pr = denormalize(p,  xmin, xmax)
    yr = denormalize(yf, xmin, xmax)
    return {
        "nmse": nmse(yf, p), "rmse": rmse(yr, pr),
        "mae":  mae(yr, pr), "mape": mape(yr, pr),
        "preds":  pr.tolist(), "actual": yr.tolist(),
        "train_history": [float(x) for x in history]
    }

all_results = {}
total_start = time.time()

for name, seed in STOCKS.items():
    t0 = time.time()
    print(f"\n[{name}]", end=" ", flush=True)

    series = generate_synthetic_stock(n=N_POINTS, seed=seed).values
    Xtr, ytr, Xte, yte, xmin, xmax = prepare(series)

    sr = {
        "series":  series.tolist(),
        "x_min":   float(xmin),
        "x_max":   float(xmax),
        "n_train": len(Xtr),
        "n_test":  len(Xte),
        "models":  {}
    }

    # QENN
    print("QENN", end="...", flush=True)
    q = ENN(ni=WINDOW, nh=5, no=1, c=0.5, lr=8e-4, seed=SEED)
    h = q.train(Xtr, ytr, epochs=EPOCHS)
    sr["models"]["QENN"] = evaluate(q, Xte, yte, xmin, xmax, h)

    # Classical ENNs
    for nh in NH_ENN:
        print(f"ENN-{nh}", end="...", flush=True)
        e = ENN(ni=WINDOW, nh=nh, no=1, lr=1e-3, seed=SEED)
        h = e.train(Xtr, ytr, epochs=EPOCHS)
        sr["models"][f"ENN-{nh}"] = evaluate(e, Xte, yte, xmin, xmax, h)

    print(f"✓  ({time.time()-t0:.1f}s)")
    all_results[name] = sr

with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n✅ Done in {time.time()-total_start:.1f}s  →  results.json saved")
print("\nNMSE Summary:")
print(f"{'Market':<14}", end="")
for m in ["QENN"] + [f"ENN-{n}" for n in NH_ENN]:
    print(f"{m:>10}", end="")
print()
for name, data in all_results.items():
    print(f"{name:<14}", end="")
    for m in ["QENN"] + [f"ENN-{n}" for n in NH_ENN]:
        print(f"{data['models'][m]['nmse']:>10.4f}", end="")
    print()