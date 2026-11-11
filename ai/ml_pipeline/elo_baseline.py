"""ML-05: Elo rating baseline.

Usage (from ml/pipeline):
    python elo_baseline.py

Reads  data/clean/matches.csv
Writes data/clean/elo_features.csv      (pre-match ratings + Elo probabilities per match)
       data/results/baseline_metrics.csv (adds an 'Elo' row)

How it works:
 1. Every team starts with a rating. After each match (in date order) the winner's
    rating goes up and the loser's goes down, by more if the result was a surprise.
 2. Before each match we record both teams' ratings (only past information is used).
 3. The rating gap is turned into home / draw / away probabilities with a small
    'ordered logit' model whose two settings are fitted on TRAINING seasons only.
 4. We evaluate on the validation season (the test season is not touched).
"""
import numpy as np
import pandas as pd

from common import (CLEAN_DIR, TRAIN_END, VAL_SEASON, load_matches, report, split_masks)

# Elo settings (tunable later, but tune on the validation season, never the test season)
K = 20             # how fast ratings move after each match
HOME_ADV = 60      # rating points added to the home team's rating
MEAN = 1500        # league-average rating
NEW_TEAM = 1450    # rating for a team seen for the first time (promoted teams are weaker)
REGRESS = 0.25     # at the start of each new season, pull ratings 25% back toward MEAN


def goal_multiplier(goal_diff):
    """Bigger wins move ratings more (same idea as World Football Elo)."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8


def run_elo(df):
    """Go through matches in date order. Returns pre-match ratings and final ratings."""
    n = len(df)
    home, away = df["HomeTeam"].to_numpy(), df["AwayTeam"].to_numpy()
    season = df["Season"].to_numpy()
    hg, ag = df["FTHG"].to_numpy(), df["FTAG"].to_numpy()
    y = df["y"].to_numpy()

    ratings, last_season = {}, {}
    home_pre, away_pre = np.zeros(n), np.zeros(n)

    for i in range(n):
        h, a, s = home[i], away[i], season[i]
        for t in (h, a):
            if t not in ratings:
                ratings[t] = NEW_TEAM
            elif last_season[t] != s:  # first match of a new season for this team
                ratings[t] = (1 - REGRESS) * ratings[t] + REGRESS * MEAN
            last_season[t] = s

        rh, ra = ratings[h], ratings[a]
        home_pre[i], away_pre[i] = rh, ra  # recorded BEFORE the match is played

        expected_home = 1 / (1 + 10 ** (-(rh + HOME_ADV - ra) / 400))
        actual_home = 1.0 if y[i] == 0 else 0.5 if y[i] == 1 else 0.0
        delta = K * goal_multiplier(hg[i] - ag[i]) * (actual_home - expected_home)
        ratings[h] = rh + delta
        ratings[a] = ra - delta

    return home_pre, away_pre, ratings


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def elo_probs(diff, scale, cut):
    """Ordered logit: rating gap -> [P(home), P(draw), P(away)]."""
    z = diff / scale
    p_away = sigmoid(-cut - z)
    p_home = sigmoid(z - cut)
    p_draw = 1 - p_away - p_home
    return np.column_stack([p_home, p_draw, p_away])


def fit_probability_model(diff_train, y_train):
    """Grid search for the scale and draw-width that minimise log loss on TRAINING data."""
    best = (None, None, np.inf)
    for scale in np.arange(100, 601, 10):
        for cut in np.arange(0.10, 1.01, 0.02):
            p = np.clip(elo_probs(diff_train, scale, cut), 1e-15, 1)
            ll = -np.mean(np.log(p[np.arange(len(y_train)), y_train]))
            if ll < best[2]:
                best = (scale, cut, ll)
    return best


def main():
    df = load_matches()
    train_m, val_m, _ = split_masks(df)

    home_pre, away_pre, final_ratings = run_elo(df)
    diff = home_pre + HOME_ADV - away_pre  # rating gap including home advantage

    scale, cut, train_ll = fit_probability_model(diff[train_m.to_numpy()],
                                                 df.loc[train_m, "y"].to_numpy())
    print(f"Fitted on seasons up to {TRAIN_END}: scale={scale:.0f}, draw width={cut:.2f} "
          f"(training log loss {train_ll:.4f})\n")

    probs = elo_probs(diff, scale, cut)

    print("RESULTS")
    print("-" * 100)
    report("Elo (train seasons)", f"<= {TRAIN_END}", probs[train_m.to_numpy()],
           df.loc[train_m, "y"].to_numpy())
    report("Elo", VAL_SEASON, probs[val_m.to_numpy()], df.loc[val_m, "y"].to_numpy())

    # Save per-match Elo values (usable later as features; all are pre-match values)
    out = df[["Div", "Date", "Season", "HomeTeam", "AwayTeam"]].copy()
    out["home_elo_pre"] = home_pre.round(1)
    out["away_elo_pre"] = away_pre.round(1)
    out["elo_diff"] = diff.round(1)
    out["elo_p_home"], out["elo_p_draw"], out["elo_p_away"] = probs.round(4).T
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_DIR / "elo_features.csv", index=False)

    # Sanity check: the strongest teams right now should look familiar
    top = sorted(final_ratings.items(), key=lambda kv: kv[1], reverse=True)[:10]
    print("\nTop 10 teams by current Elo (sanity check):")
    for name, rating in top:
        print(f"  {name:<28} {rating:.0f}")
    print(f"\nSaved {CLEAN_DIR / 'elo_features.csv'}")


if __name__ == "__main__":
    main()
