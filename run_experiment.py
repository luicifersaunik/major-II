"""
run_experiment_fast.py — Fair Comparison: QENN-Qiskit vs Classical ENN
Both use Adam optimizer. ENN uses per-size learning rates for stability.

Models:
  1. QENN-Qiskit  — Qiskit, 5 qubits, CNOT, Adam + SPSA
  2. ENN-10/20/40/70/100 — Adam optimizer, Xavier init, per-size lr

Setup:
    pip install numpy pandas qiskit

Run:
    python run_experiment_fast.py
"""

import json
import time
import numpy as np

from data_utils import (
    generate_synthetic_stock,
    normalize, denormalize,
    sliding_window, train_test_split_sequential,
    nmse, rmse, mae, mape
)
from enn_baseline import ENN

# Check Qiskit
try:
    from qenn_qiskit import QENN_Qiskit
    QISKIT_AVAILABLE = True
    print("Qiskit found — QENN-Qiskit will be included")
except ImportError:
    QISKIT_AVAILABLE = False
    print("Qiskit not found — run: pip install qiskit")

# ── Configuration ──────────────────────────────────────────────────────────
WINDOW        = 6
EPOCHS        = 80     # same for all models — fair comparison
QISKIT_EPOCHS = 80
QISKIT_NH     = 5
SEED          = 42
NH_ENN        = [10, 20, 40, 70, 100]
STOCKS        = ["NIFTY", "NASDAQ", "HSI", "SSE", "Russell2000", "TAIEX"]
SEEDS         = {
    "NIFTY": 1, "NASDAQ": 2, "HSI": 3,
    "SSE": 4, "Russell2000": 5, "TAIEX": 6
}

# ── Helpers ────────────────────────────────────────────────────────────────
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
        "nmse":          nmse(yf, p),
        "rmse":          rmse(yr, pr),
        "mae":           mae(yr, pr),
        "mape":          mape(yr, pr),
        "preds":         [round(float(x), 2) for x in pr],
        "actual":        [round(float(x), 2) for x in yr],
        "train_history": [round(float(x), 6) for x in history],
    }

# ── Main ───────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  FAIR Comparison: QENN-Qiskit vs Classical ENN")
print(f"  Both use Adam optimizer + Xavier init + gradient clipping")
print(f"  Data  : Synthetic GBM (400 points, 6 markets)")
print(f"  Epochs: {EPOCHS} for all models")
print("="*65)

all_results = {}
total_start = time.time()

for name in STOCKS:
    print(f"\n[{name}]")
    t0 = time.time()

    series = generate_synthetic_stock(n=400, seed=SEEDS[name]).values
    Xtr, ytr, Xte, yte, xmin, xmax = prepare(series)
    print(f"  Train={len(Xtr)}  Test={len(Xte)}")

    sr = {
        "series":      [round(float(x), 2) for x in series[-80:]],
        "x_min":       float(xmin),
        "x_max":       float(xmax),
        "n_train":     len(Xtr),
        "n_test":      len(Xte),
        "data_source": "synthetic (GBM)",
        "models":      {}
    }

    # ── QENN-Qiskit ───────────────────────────────────────────────
    if QISKIT_AVAILABLE:
        print(f"  QENN-Qiskit  (5 qubits, Adam+SPSA)...", end=" ", flush=True)
        t1 = time.time()
        qk = QENN_Qiskit(ni=WINDOW, nh=QISKIT_NH, no=1, c=0.5, seed=SEED)
        h  = qk.train(Xtr, ytr, epochs=QISKIT_EPOCHS)
        sr["models"]["QENN-Qiskit"] = evaluate(qk, Xte, yte, xmin, xmax, h)
        sr["models"]["QENN-Qiskit"]["time"] = round(time.time()-t1, 1)
        print(f"NMSE={sr['models']['QENN-Qiskit']['nmse']:.4f}  "
              f"({sr['models']['QENN-Qiskit']['time']}s)")

    # ── Classical ENN (Adam, per-size lr) ─────────────────────────
    for nh in NH_ENN:
        print(f"  ENN-{nh:<3}      (Adam, auto-lr)...", end=" ", flush=True)
        t1 = time.time()
        # lr=None → ENN auto-selects stable lr for this network size
        e = ENN(ni=WINDOW, nh=nh, no=1, c=0.0, lr=None, seed=SEED)
        h = e.train(Xtr, ytr, epochs=EPOCHS)
        k = f"ENN-{nh}"
        sr["models"][k] = evaluate(e, Xte, yte, xmin, xmax, h)
        sr["models"][k]["time"] = round(time.time()-t1, 1)
        print(f"NMSE={sr['models'][k]['nmse']:.4f}  "
              f"({sr['models'][k]['time']}s)")

    print(f"  [{name} done in {time.time()-t0:.1f}s]")
    all_results[name] = sr

# ── Save ───────────────────────────────────────────────────────────────────
with open("results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*65}")
print(f"  Total: {time.time()-total_start:.1f}s  →  results.json saved")
print(f"  Next : copy results.json qenn-dashboard\\public\\results.json")
print(f"{'='*65}")

# ── NMSE Summary ──────────────────────────────────────────────────────────
models = (["QENN-Qiskit"] if QISKIT_AVAILABLE else []) + \
         [f"ENN-{n}" for n in NH_ENN]

print(f"\nFAIR NMSE Summary (* = best per market):\n")
print(f"{'Market':<14}", end="")
for m in models:
    print(f"{m:>13}", end="")
print()
print("-" * (14 + 13*len(models)))

for name, data in all_results.items():
    vals  = {m: data["models"].get(m, {}).get("nmse") for m in models}
    valid = [v for v in vals.values() if v is not None]
    best  = min(valid) if valid else None
    print(f"{name:<14}", end="")
    for m in models:
        v = vals[m]
        if v is None:
            print(f"{'—':>13}", end="")
        elif v == best:
            print(f"{'*'+f'{v:.4f}':>13}", end="")
        else:
            print(f"{v:>13.4f}", end="")
    print()

print("\n* = best NMSE for that market")
print("\nFair comparison: all models use Adam optimizer + Xavier init + gradient clipping")