"""Shared helpers for the Scoreline ML scripts (data loading, splits, metrics).

Run the other scripts from the same folder as this file, e.g.
    cd ml/pipeline
    python eda_benchmarks.py
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path(os.getenv("CLEAN_DATA_DIR", "data/clean"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "data/results"))

# Time-based split (never split randomly!)
TRAIN_END = "2023/24"   # training data: this season and everything earlier
VAL_SEASON = "2024/25"  # used to compare models and tune
TEST_SEASON = "2025/26" # final test: do NOT look at this until the very end

OUTCOME_CODES = {"H": 0, "D": 1, "A": 2}  # probability columns are always [home, draw, away]


def load_matches():
    path = CLEAN_DIR / "matches.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run clean.py first (from the same folder).")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values(["Date", "Div", "HomeTeam"]).reset_index(drop=True)
    df["y"] = df["FTR"].map(OUTCOME_CODES)
    return df


def split_masks(df):
    """Boolean masks for train / validation / test rows. Seasons are 'YYYY/YY' strings."""
    train = df["Season"] <= TRAIN_END
    val = df["Season"] == VAL_SEASON
    test = df["Season"] == TEST_SEASON
    return train, val, test


# ---------- Metrics (p = array of shape [n, 3] with columns home, draw, away) ----------

def log_loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier(p, y):
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((np.asarray(p) - onehot) ** 2, axis=1)))


def rps(p, y):
    """Ranked Probability Score: respects that home / draw / away are ordered."""
    onehot = np.eye(3)[y]
    cum_p = np.cumsum(p, axis=1)[:, :2]
    cum_y = np.cumsum(onehot, axis=1)[:, :2]
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / 2))


def accuracy(p, y):
    return float(np.mean(np.argmax(p, axis=1) == y))


def report(model, season, p, y):
    """Print one results line and store it in data/results/baseline_metrics.csv."""
    row = {
        "model": model, "season": season, "n_matches": len(y),
        "log_loss": round(log_loss(p, y), 4), "brier": round(brier(p, y), 4),
        "rps": round(rps(p, y), 4), "accuracy": round(accuracy(p, y), 4),
    }
    print(f"{model:<28} {season}  n={row['n_matches']:<5} log_loss={row['log_loss']:.4f}  "
          f"brier={row['brier']:.4f}  rps={row['rps']:.4f}  accuracy={row['accuracy']:.1%}")
    save_metrics(row)
    return row


def save_metrics(row):
    """Add a results row, replacing any earlier row for the same model and season."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "baseline_metrics.csv"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        old = old[~((old["model"] == row["model"]) & (old["season"] == row["season"]))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
