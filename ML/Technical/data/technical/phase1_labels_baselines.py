"""
Phase 1 helper: label distribution + baselines for the Buy/Hold/Sell project.

Run it on your own machine, where all 10 technical files already exist.
It reads data/technical/<TICKER>/technical_featured_unscaled.csv for every
ticker, and writes ONE small file (phase1_summary.json) plus a printed report.
Upload or paste that summary instead of the big CSVs.

Usage:
    python phase1_labels_baselines.py --data-dir data/technical --theta 0.0075

Needs: pandas, numpy, scikit-learn.

Label rule (same as the Phase 1 Performa):
    r = Close(t+1) / Close(t) - 1
    Buy if r > +theta, Sell if r < -theta, else Hold.

Splits follow the pipeline: chronological 70 / 15 / 15 by rows, per stock,
30-day windows inside each split (so sequences = split rows - 30).
Baselines use the 27 technical features of the last day of the window (day t).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

CLASSES = ["Buy", "Hold", "Sell"]
WINDOW = 30
FILE_NAME = "technical_featured_unscaled.csv"


def label(r, theta):
    return "Buy" if r > theta else ("Sell" if r < -theta else "Hold")


def split_rows(n):
    a, b = int(0.70 * n), int(0.85 * n)
    return {"train": (0, a), "val": (a, b), "test": (b, n)}


def samples_for_split(df, lo, hi, theta, feat_cols):
    """Rows i = lo+WINDOW-1 .. hi-2 (needs next-day close inside the split)."""
    seg = df.iloc[lo:hi].reset_index(drop=True)
    idx = np.arange(WINDOW - 1, len(seg) - 1)
    close = seg["close"].to_numpy()
    r = close[idx + 1] / close[idx] - 1
    y = np.array([label(x, theta) for x in r])
    X = seg.loc[idx, feat_cols].to_numpy(dtype=float)
    return X, y, r


def props(y):
    n = len(y)
    return {c: round(float((y == c).sum()) / n, 4) if n else None for c in CLASSES}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/technical")
    ap.add_argument("--theta", type=float, default=0.0075)
    ap.add_argument("--out", default="phase1_summary.json")
    args = ap.parse_args()

    root = Path(args.data_dir)
    files = sorted(root.glob(f"*/{FILE_NAME}"))
    if not files:
        sys.exit(f"No {FILE_NAME} found under {root}/<TICKER>/")

    per_stock, pooled = {}, {s: {"X": [], "y": [], "r": []} for s in ("train", "val", "test")}
    for f in files:
        ticker = f.parent.name
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        feat_cols = [c for c in df.columns if c != "date"]
        info = {
            "rows": int(len(df)),
            "first_date": str(df["date"].iloc[0].date()),
            "last_date": str(df["date"].iloc[-1].date()),
            "n_features": len(feat_cols),
            "nulls": int(df.isna().sum().sum()),
            "duplicate_dates": int(df["date"].duplicated().sum()),
        }
        scaler = StandardScaler()
        for name, (lo, hi) in split_rows(len(df)).items():
            X, y, r = samples_for_split(df, lo, hi, args.theta, feat_cols)
            if name == "train":
                scaler.fit(X)
                info["train_next_day_return"] = {
                    "mean": round(float(r.mean()), 5), "sd": round(float(r.std()), 5),
                    "min": round(float(r.min()), 4), "max": round(float(r.max()), 4)}
            info[f"{name}_sequences"] = int(len(y))
            info[f"{name}_class_proportions"] = props(y)
            pooled[name]["X"].append(scaler.transform(X))  # per-stock scaling, fit on that stock's train
            pooled[name]["y"].append(y)
            pooled[name]["r"].append(r)
        per_stock[ticker] = info

    Xtr, ytr = np.vstack(pooled["train"]["X"]), np.concatenate(pooled["train"]["y"])
    Xte, yte = np.vstack(pooled["test"]["X"]), np.concatenate(pooled["test"]["y"])

    # theta sweep on pooled train returns (use to confirm theta)
    rtr = np.concatenate(pooled["train"]["r"])
    sweep = {}
    for th in (0.003, 0.005, 0.0075, 0.01, 0.0125):
        yy = np.array([label(x, th) for x in rtr])
        sweep[str(th)] = props(yy)

    def scores(pred):
        return {
            "macro_f1": round(float(f1_score(yte, pred, labels=CLASSES, average="macro", zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(yte, pred)), 4),
            "recall_per_class": {c: round(float(v), 4) for c, v in zip(
                CLASSES, recall_score(yte, pred, labels=CLASSES, average=None, zero_division=0))},
        }

    majority = pd.Series(ytr).mode()[0]
    base_majority = scores(np.full(len(yte), majority))
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    base_lr = scores(lr.predict(Xte))

    summary = {
        "theta": args.theta,
        "window": WINDOW,
        "stocks": per_stock,
        "pooled": {
            "train_sequences": int(len(ytr)),
            "val_sequences": int(sum(len(a) for a in pooled["val"]["y"])),
            "test_sequences": int(len(yte)),
            "train_class_proportions": props(ytr),
            "test_class_proportions": props(yte),
            "theta_sweep_train_proportions": sweep,
        },
        "baselines_on_pooled_test": {
            "majority_class": {"predicts": majority, **base_majority},
            "logistic_regression_technical_only": base_lr,
        },
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    # readable report
    print(f"theta = {args.theta}   stocks found: {len(per_stock)}")
    print(f"{'ticker':<14}{'rows':>6}{'train/val/test':>16}   train Buy/Hold/Sell")
    for t, i in per_stock.items():
        p = i["train_class_proportions"]
        print(f"{t:<14}{i['rows']:>6}{i['train_sequences']:>7}/{i['val_sequences']}/{i['test_sequences']:<5}"
              f"   {p['Buy']:.3f} / {p['Hold']:.3f} / {p['Sell']:.3f}")
    p = summary["pooled"]
    print(f"\nPOOLED sequences train/val/test: {p['train_sequences']}/{p['val_sequences']}/{p['test_sequences']}")
    print("POOLED train class proportions:", p["train_class_proportions"])
    print("Theta sweep (pooled train):")
    for th, v in sweep.items():
        print(f"  {th}: {v}")
    print("\nBaselines on pooled test set:")
    for name, v in summary["baselines_on_pooled_test"].items():
        print(f"  {name}: {v}")
    print(f"\nWrote {args.out} (small; upload this instead of the CSVs).")


if __name__ == "__main__":
    main()