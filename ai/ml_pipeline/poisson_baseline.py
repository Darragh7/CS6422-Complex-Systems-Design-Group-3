"""ML-06: Poisson goals baseline.

Usage (from ml/pipeline):
    python poisson_baseline.py            # run with the default settings
    python poisson_baseline.py --tune     # try a few settings on the validation season

Reads  data/clean/matches.csv  (and data/clean/elo_features.csv if it exists)
Writes data/clean/poisson_predictions.csv   (expected goals + probabilities, validation season)
       data/results/baseline_metrics.csv    (adds 'Poisson' rows)

How it works:
 1. Each team gets an ATTACK strength and a DEFENCE strength, plus one shared HOME advantage.
 2. Expected goals for the home team = exp(home advantage + home attack - away defence);
    expected goals for the away team = exp(away attack - home defence).
 3. Goals are modelled as Poisson numbers with those averages. Combining both teams'
    Poisson distributions gives the probability of every scoreline (0-0, 1-0, ...), and
    adding them up gives P(home win), P(draw), P(away win).
 4. Team strengths change over time, so the model is re-fitted every week using only
    PAST matches (the last 3 years, with recent matches weighted more) - a walk-forward test.
"""
import argparse
import itertools
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import PoissonRegressor

from common import CLEAN_DIR, VAL_SEASON, load_matches, report, split_masks

# Settings (tune on the validation season only, never on the test season)
WINDOW_DAYS = 1095      # only use matches from the last 3 years
HALF_LIFE_DAYS = 365    # a match from 1 year ago counts half as much as one from today
ALPHA = 0.001           # small penalty that keeps team strengths sensible (esp. promoted teams)
REFIT_EVERY_DAYS = 7    # re-fit the model once a week
MAX_GOALS = 10          # scorelines up to 10-10 are considered


def build_design(df):
    """One row per team per match: [home flag | attack columns | defence columns]."""
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    idx = {t: i for i, t in enumerate(teams)}
    T, n = len(teams), len(df)
    h = df["HomeTeam"].map(idx).to_numpy()
    a = df["AwayTeam"].map(idx).to_numpy()

    X = np.zeros((2 * n, 1 + 2 * T), dtype=np.float32)
    rows_home = np.arange(0, 2 * n, 2)  # row that predicts the home team's goals
    rows_away = rows_home + 1           # row that predicts the away team's goals
    X[rows_home, 0] = 1                 # home advantage applies to the home team only
    X[rows_home, 1 + h] = 1             # home team's attack
    X[rows_home, 1 + T + a] = 1         # away team's defence
    X[rows_away, 1 + a] = 1             # away team's attack
    X[rows_away, 1 + T + h] = 1         # home team's defence

    y = np.empty(2 * n)
    y[rows_home] = df["FTHG"].to_numpy()
    y[rows_away] = df["FTAG"].to_numpy()
    return X, y, teams


def fit_window(X, y, day_row, start_day, alpha, half_life, window):
    """Fit on matches strictly BEFORE start_day (no future information)."""
    m = (day_row < start_day) & (day_row >= start_day - window)
    weights = 0.5 ** ((start_day - day_row[m]) / half_life)
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X[m].astype(np.float64), y[m], sample_weight=weights)
    n_warn = sum(issubclass(c.category, ConvergenceWarning) for c in caught)
    return model, n_warn


def poisson_pmf(lam, kmax):
    """P(0..kmax goals) for each expected-goals value in lam."""
    out = np.zeros((len(lam), kmax + 1))
    out[:, 0] = np.exp(-lam)
    for k in range(1, kmax + 1):
        out[:, k] = out[:, k - 1] * lam / k
    return out


def outcome_probs(lam_home, lam_away):
    """Expected goals -> [P(home win), P(draw), P(away win)] via the scoreline grid."""
    ph = poisson_pmf(lam_home, MAX_GOALS)
    pa = poisson_pmf(lam_away, MAX_GOALS)
    grid = ph[:, :, None] * pa[:, None, :]  # grid[match, home_goals, away_goals]
    goals = np.arange(MAX_GOALS + 1)
    home_win = goals[:, None] > goals[None, :]
    away_win = goals[:, None] < goals[None, :]
    p_home = (grid * home_win).sum(axis=(1, 2))
    p_away = (grid * away_win).sum(axis=(1, 2))
    p_draw = (grid * ~(home_win | away_win)).sum(axis=(1, 2))
    probs = np.column_stack([p_home, p_draw, p_away])
    return probs / probs.sum(axis=1, keepdims=True)


def walk_forward(X, y, day_match, val_idx, alpha, half_life, window=WINDOW_DAYS):
    """Predict each validation match using a model fitted only on earlier matches."""
    day_row = np.repeat(day_match, 2)
    val_days = day_match[val_idx]
    lam = np.zeros((len(val_idx), 2))
    warn_total, fits = 0, 0
    start, end = val_days.min(), val_days.max()
    while start <= end:
        in_block = (val_days >= start) & (val_days < start + REFIT_EVERY_DAYS)
        if in_block.any():
            model, n_warn = fit_window(X, y, day_row, start, alpha, half_life, window)
            warn_total, fits = warn_total + n_warn, fits + 1
            m_idx = val_idx[in_block]
            rows = np.column_stack([2 * m_idx, 2 * m_idx + 1]).ravel()
            lam[in_block] = model.predict(X[rows].astype(np.float64)).reshape(-1, 2)
        start += REFIT_EVERY_DAYS
    return lam, warn_total, fits


def main():
    parser = argparse.ArgumentParser(description="Poisson goals baseline")
    parser.add_argument("--tune", action="store_true", help="try a few settings on the validation season")
    args = parser.parse_args()

    df = load_matches()
    _, val_m, _ = split_masks(df)
    val_idx = np.flatnonzero(val_m.to_numpy())
    y_val = df.loc[val_m, "y"].to_numpy()

    X, y, teams = build_design(df)
    day_match = (df["Date"] - df["Date"].min()).dt.days.to_numpy()

    if args.tune:
        print(f"Trying settings on {VAL_SEASON} (this takes a few minutes)...\n")
        print(f"{'half_life':>10} {'alpha':>8} {'log_loss':>9} {'accuracy':>9}")
        for half_life, alpha in itertools.product([180, 365, 730], [0.001, 0.003, 0.01]):
            lam, _, _ = walk_forward(X, y, day_match, val_idx, alpha, half_life)
            p = outcome_probs(lam[:, 0], lam[:, 1])
            ll = -np.mean(np.log(np.clip(p[np.arange(len(y_val)), y_val], 1e-15, 1)))
            acc = np.mean(p.argmax(axis=1) == y_val)
            print(f"{half_life:>10} {alpha:>8} {ll:>9.4f} {acc:>9.1%}")
        print("\nTo use a setting, change HALF_LIFE_DAYS and ALPHA at the top of this file.")
        print("These numbers are slightly optimistic because we tuned on this season;")
        print("the untouched test season gives the honest final number later.")
        return

    t0 = time.time()
    lam, n_warn, fits = walk_forward(X, y, day_match, val_idx, ALPHA, HALF_LIFE_DAYS)
    print(f"Walk-forward: {fits} weekly re-fits in {time.time() - t0:.0f}s "
          f"(convergence warnings: {n_warn})\n")
    probs = outcome_probs(lam[:, 0], lam[:, 1])

    print("RESULTS")
    print("-" * 100)
    report("Poisson", VAL_SEASON, probs, y_val)

    # Quick extra: average Poisson with Elo if the Elo file exists
    elo_path = CLEAN_DIR / "elo_features.csv"
    if elo_path.exists():
        elo = pd.read_csv(elo_path, parse_dates=["Date"])
        keys = ["Date", "HomeTeam", "AwayTeam"]
        merged = df.loc[val_m, keys].merge(elo[keys + ["elo_p_home", "elo_p_draw", "elo_p_away"]],
                                           on=keys, how="left")
        elo_p = merged[["elo_p_home", "elo_p_draw", "elo_p_away"]].to_numpy()
        if len(merged) == len(y_val) and not np.isnan(elo_p).any():
            report("Elo + Poisson (average)", VAL_SEASON, (elo_p + probs) / 2, y_val)
        else:
            print("(Skipped the Elo + Poisson average: could not match all matches.)")
    else:
        print("(Run elo_baseline.py first to also get an Elo + Poisson average.)")

    # Save predictions
    out = df.loc[val_m, ["Div", "Date", "Season", "HomeTeam", "AwayTeam"]].copy()
    out["exp_home_goals"], out["exp_away_goals"] = lam[:, 0].round(3), lam[:, 1].round(3)
    out["poisson_p_home"], out["poisson_p_draw"], out["poisson_p_away"] = probs.round(4).T
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_DIR / "poisson_predictions.csv", index=False)

    # Sanity check: fit on the latest data and show the strongest teams
    last_day = day_match.max() + 1
    model, _ = fit_window(X, y, np.repeat(day_match, 2), last_day, ALPHA, HALF_LIFE_DAYS, WINDOW_DAYS)
    T = len(teams)
    attack, defence = model.coef_[1:1 + T], -model.coef_[1 + T:]  # higher defence = concedes less
    recent = set(df.loc[day_match > day_match.max() - 365, "HomeTeam"]) | \
             set(df.loc[day_match > day_match.max() - 365, "AwayTeam"])
    strengths = pd.DataFrame({"team": teams, "attack": attack, "defence": defence})
    strengths = strengths[strengths["team"].isin(recent)]
    print(f"\nHome advantage: home teams score about x{np.exp(model.coef_[0]):.2f} "
          f"more goals than they otherwise would")
    print("\nBest attacks right now (sanity check):")
    print(strengths.nlargest(10, "attack")[["team", "attack"]].round(2).to_string(index=False))
    print("\nBest defences right now (sanity check):")
    print(strengths.nlargest(10, "defence")[["team", "defence"]].round(2).to_string(index=False))
    print(f"\nSaved {CLEAN_DIR / 'poisson_predictions.csv'}")


if __name__ == "__main__":
    main()
