"""ML-04: quick exploratory analysis + the 'dumb' benchmarks every model must beat.

Usage (from ml/pipeline):
    python eda_benchmarks.py

Reads  data/clean/matches.csv
Writes plots to data/plots/ and benchmark results to data/results/baseline_metrics.csv
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (VAL_SEASON, TEST_SEASON, TRAIN_END, load_matches, report,
                    save_metrics, split_masks)

PLOTS = Path("data/plots")
PLOTS.mkdir(parents=True, exist_ok=True)

df = load_matches()
train_m, val_m, test_m = split_masks(df)

print("=" * 70)
print("DATA SIZE")
print("=" * 70)
print(f"All matches: {len(df)}")
print(f"Train (up to {TRAIN_END}): {train_m.sum()} | Validation ({VAL_SEASON}): {val_m.sum()} "
      f"| Test ({TEST_SEASON}): {test_m.sum()} (do not use yet)")
print(f"Newer matches not in any split (current season so far): {(~(train_m | val_m | test_m)).sum()}")

# ---------------------------------------------------------------- 1. results
print("\n" + "=" * 70)
print("1. RESULT SPLIT (% of matches)")
print("=" * 70)
overall = df["FTR"].value_counts(normalize=True).reindex(["H", "D", "A"]) * 100
print("Overall:", overall.round(1).to_dict())
by_league = (df.groupby("League")["FTR"].value_counts(normalize=True).unstack()
             .reindex(columns=["H", "D", "A"]) * 100).round(1)
print("\nBy league:")
print(by_league)

by_league.plot(kind="bar", figsize=(8, 4))
plt.title("Result split by league (%)")
plt.ylabel("% of matches")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(PLOTS / "result_split_by_league.png", dpi=120)
plt.close()

# ----------------------------------------------------- 2. home advantage by season
print("\n" + "=" * 70)
print("2. HOME ADVANTAGE BY SEASON")
print("=" * 70)
by_season = df.groupby("Season").agg(
    matches=("FTR", "size"),
    home_win_pct=("FTR", lambda s: (s == "H").mean() * 100),
    draw_pct=("FTR", lambda s: (s == "D").mean() * 100),
    avg_home_goals=("FTHG", "mean"),
    avg_away_goals=("FTAG", "mean"),
).round(2)
print(by_season)
print("\nNote: 2019/20 and 2020/21 had many matches played without fans, "
      "so home advantage dips there.")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(by_season.index, by_season["home_win_pct"], marker="o")
ax.set_title("Home win % by season")
ax.set_ylabel("% home wins")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(PLOTS / "home_win_pct_by_season.png", dpi=120)
plt.close()

# ----------------------------------------------------- 3. goals vs Poisson
print("\n" + "=" * 70)
print("3. GOALS PER TEAM PER MATCH vs POISSON")
print("=" * 70)
all_goals = pd.concat([df["FTHG"], df["FTAG"]]).astype(int)
mean_goals = all_goals.mean()
max_k = 7
observed = [(all_goals == k).mean() for k in range(max_k + 1)]
poisson = [math.exp(-mean_goals) * mean_goals ** k / math.factorial(k) for k in range(max_k + 1)]
cmp = pd.DataFrame({"goals": range(max_k + 1), "observed": np.round(observed, 3),
                    "poisson": np.round(poisson, 3)})
print(f"Average goals per team per match: {mean_goals:.2f}")
print(cmp.to_string(index=False))

x = np.arange(max_k + 1)
plt.figure(figsize=(8, 4))
plt.bar(x - 0.2, observed, width=0.4, label="Observed")
plt.bar(x + 0.2, poisson, width=0.4, label="Poisson")
plt.title("Goals per team per match: observed vs Poisson")
plt.xlabel("Goals")
plt.ylabel("Share of team-matches")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "goals_vs_poisson.png", dpi=120)
plt.close()

# ----------------------------------------------------- 4. league differences
print("\n" + "=" * 70)
print("4. LEAGUE DIFFERENCES")
print("=" * 70)
league_stats = df.groupby("League").agg(
    avg_total_goals=("FTHG", lambda s: s.mean() + df.loc[s.index, "FTAG"].mean()),
    draw_pct=("FTR", lambda s: (s == "D").mean() * 100),
).round(2)
print(league_stats)

# ----------------------------------------------------- 5. dumb benchmarks
print("\n" + "=" * 70)
print(f"5. BENCHMARKS on validation season {VAL_SEASON} (any real model must beat these)")
print("=" * 70)
y_val = df.loc[val_m, "y"].to_numpy()
y_train = df.loc[train_m, "y"].to_numpy()

# a) always predict a home win (accuracy only, since it gives no probabilities)
acc_home = float(np.mean(y_val == 0))
print(f"{'Always predict home win':<28} {VAL_SEASON}  accuracy={acc_home:.1%}")
save_metrics({"model": "Always home win", "season": VAL_SEASON, "n_matches": len(y_val),
              "log_loss": np.nan, "brier": np.nan, "rps": np.nan, "accuracy": round(acc_home, 4)})

# b) predict the historical frequencies for every match
freq = np.bincount(y_train, minlength=3) / len(y_train)
print(f"\nTraining-set frequencies [home, draw, away]: {np.round(freq, 3)}")
p_freq = np.tile(freq, (len(y_val), 1))
report("Historical frequencies", VAL_SEASON, p_freq, y_val)

print("\nPlots saved in:", PLOTS.resolve())
print("Results saved in: data/results/baseline_metrics.csv")
