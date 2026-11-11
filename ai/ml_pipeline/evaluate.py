"""ML-09 / ML-10: honest evaluation, bookmaker comparison and calibration check.

Usage (from ml/pipeline):
    python evaluate.py
    python evaluate.py --model "Logistic regression"     # calibration for another model

Run first: elo_baseline.py, poisson_baseline.py, features.py, train_models.py, odds.py

Reads  the validation-season predictions of every model + data/clean/odds.csv
Writes data/results/evaluation_val.csv        (overall table)
       data/plots/calibration_<model>.png      (calibration plot)

It answers four questions:
 1. Which model is best overall?                 (metrics table)
 2. Is it better in some leagues?                (per-league log loss)
 3. Are the differences REAL or just noise?      (bootstrap confidence intervals)
 4. Do '60%' predictions really happen ~60%?     (calibration)
Only the validation season is used. The test season stays untouched.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CLEAN_DIR, RESULTS_DIR, VAL_SEASON, accuracy, brier, load_matches,
                    log_loss, rps)

KEYS = ["Date", "HomeTeam", "AwayTeam"]
NICE = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "gradient_boosting_sklearn": "Gradient boosting (sklearn)",
    "lr_and_xgboost_average": "LR + XGBoost (average)",
    "lr_and_gradient_boosting_sklearn_average": "LR + gradient boosting (average)",
}
ENSEMBLE = "Ensemble (Elo + Poisson + LR + boosting)"
MARKET = "Bookmakers (margin removed)"
OUTCOMES = ["Home win", "Draw", "Away win"]
N_BOOT = 2000


def probs_from(df, prefix):
    return df[[f"{prefix}_p_home", f"{prefix}_p_draw", f"{prefix}_p_away"]].to_numpy()


def load_everything():
    base = load_matches()
    base = base[base["Season"] == VAL_SEASON][["Div", "League"] + KEYS + ["y"]].reset_index(drop=True)

    def need(name):
        path = CLEAN_DIR / name
        if not path.exists():
            raise SystemExit(f"{path} not found. Run the earlier scripts first (see top of this file).")
        return pd.read_csv(path, parse_dates=["Date"])

    elo = need("elo_features.csv")[KEYS + ["elo_p_home", "elo_p_draw", "elo_p_away"]]
    poi = need("poisson_predictions.csv")[KEYS + ["poisson_p_home", "poisson_p_draw", "poisson_p_away"]]
    mod = need("model_predictions_val.csv").drop(columns=["Div", "Season"])

    df = base.merge(elo, on=KEYS, how="left").merge(poi, on=KEYS, how="left").merge(mod, on=KEYS, how="left")
    if df.isna().any().any():
        raise SystemExit("Some validation matches are missing predictions. Re-run the earlier scripts.")

    models = {"Elo": probs_from(df, "elo"), "Poisson": probs_from(df, "poisson")}
    tags = sorted({c[:-len("_p_home")] for c in df.columns if c.endswith("_p_home")})
    for tag in tags:
        if tag in ("elo", "poisson") or tag.startswith("market"):
            continue
        models[NICE.get(tag, re.sub("_", " ", tag).capitalize())] = probs_from(df, tag)

    boosting = models.get("XGBoost", models.get("Gradient boosting (sklearn)"))
    models[ENSEMBLE] = (models["Elo"] + models["Poisson"] + models["Logistic regression"] + boosting) / 4

    odds_path = CLEAN_DIR / "odds.csv"
    has_odds = np.ones(len(df), dtype=bool)
    if odds_path.exists():
        odds = pd.read_csv(odds_path, parse_dates=["Date"])[KEYS + ["market_p_home", "market_p_draw", "market_p_away"]]
        df = df.merge(odds, on=KEYS, how="left")
        has_odds = df["market_p_home"].notna().to_numpy()
        models[MARKET] = np.nan_to_num(probs_from(df, "market"), nan=1 / 3)
        print(f"Bookmaker odds found for {has_odds.sum()} of {len(df)} validation matches.")
    else:
        print("(No odds.csv found. Run odds.py to add the bookmaker comparison.)")
    return df, models, has_odds


def per_match_loss(p, y):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1))


def bootstrap_ci(diff, rng):
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def calibration(p, y, name):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    print(f"\nCALIBRATION of '{name}' (quantile bins, about {len(y) // 8} matches each)")
    print("Gaps under about 5 points are just noise at this sample size.\n")
    for k, ax in enumerate(axes):
        pk, hit = p[:, k], (y == k).astype(float)
        bins = pd.qcut(pk, q=8, duplicates="drop")
        t = pd.DataFrame({"pred": pk, "hit": hit, "bin": bins}).groupby("bin", observed=True).agg(
            n=("hit", "size"), predicted=("pred", "mean"), observed=("hit", "mean"))
        ece = float((t["n"] * (t["observed"] - t["predicted"]).abs()).sum() / t["n"].sum())
        print(f"{OUTCOMES[k]}: average gap between predicted and observed = {ece * 100:.1f} points")
        print(t.assign(predicted=t["predicted"].round(3), observed=t["observed"].round(3))
              .reset_index(drop=True).to_string(index=False))
        print()
        lim = max(t["predicted"].max(), t["observed"].max()) * 1.1
        ax.plot([0, lim], [0, lim], "--", color="grey")
        ax.plot(t["predicted"], t["observed"], "o-")
        ax.set_title(f"{OUTCOMES[k]} (avg gap {ece * 100:.1f} pts)")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
    fig.suptitle(f"Calibration: {name}")
    plt.tight_layout()
    Path("data/plots").mkdir(parents=True, exist_ok=True)
    out = Path("data/plots") / f"calibration_{re.sub('[^a-z0-9]+', '_', name.lower()).strip('_')}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ENSEMBLE, help="model to check for calibration")
    args = parser.parse_args()

    df, models, has_odds = load_everything()
    y_all = df["y"].to_numpy().astype(int)
    sub = has_odds
    y = y_all[sub]
    print(f"\nEvaluating on {VAL_SEASON}: {sub.sum()} matches"
          f"{' (only matches that have bookmaker odds, so every row is comparable)' if sub.sum() < len(df) else ''}")

    # 1. Overall table
    rows = []
    for name, p in models.items():
        ps = p[sub]
        rows.append({"model": name, "n": len(y), "log_loss": log_loss(ps, y), "brier": brier(ps, y),
                     "rps": rps(ps, y), "accuracy": accuracy(ps, y)})
    table = pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)
    print("\n1. OVERALL (lower is better, except accuracy)")
    print(table.round(4).to_string(index=False))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(RESULTS_DIR / "evaluation_val.csv", index=False)

    # 2. Per league
    leagues = df.loc[sub, "League"].to_numpy()
    per = {name: {lg: log_loss(p[sub][leagues == lg], y[leagues == lg]) for lg in sorted(set(leagues))}
           for name, p in models.items()}
    print("\n2. LOG LOSS BY LEAGUE")
    print(pd.DataFrame(per).T.loc[table["model"]].round(4).to_string())

    # 3. Are differences real? Paired bootstrap on per-match log loss
    rng = np.random.default_rng(0)
    losses = {name: per_match_loss(p[sub], y) for name, p in models.items()}
    print("\n3. ARE THE DIFFERENCES REAL? (average log-loss difference with 95% interval)")
    print("   Negative = better than the reference. If the interval contains 0, the models are statistically tied.\n")
    refs = ["Elo"] + ([MARKET] if MARKET in models else [])
    for ref in refs:
        for name in models:
            if name == ref:
                continue
            mean, lo, hi = bootstrap_ci(losses[name] - losses[ref], rng)
            verdict = "better" if hi < 0 else ("worse" if lo > 0 else "tied")
            print(f"   {name:<42} vs {ref:<28} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")
        print()

    # 4. Calibration
    if args.model not in models:
        raise SystemExit(f"Unknown model '{args.model}'. Choose from: {', '.join(models)}")
    calibration(models[args.model][sub], y, args.model)
    print(f"\nSaved {RESULTS_DIR / 'evaluation_val.csv'}")
    print("The test season has not been touched.")


if __name__ == "__main__":
    main()
