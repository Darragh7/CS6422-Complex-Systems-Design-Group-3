"""ML-07a / ML-07b: build model features from past matches only.

Usage (from ml/pipeline):
    python features.py

Reads  data/clean/matches.csv and data/clean/elo_features.csv (run elo_baseline.py first)
Writes data/clean/features.csv   (one row per match: identifiers + features + target 'y')

Every feature for a match is computed ONLY from matches played BEFORE it
(rolling windows are shifted by one match), and a built-in test re-checks this by hand.

Feature groups (suffix _h = home team, _a = away team, _diff = home minus away):
  form           pts5, gf5, ga5, pts10, gf10, ga10   (points / goals for / goals against per match)
  shots          sf5, sa5, sotf5, sota5              (shots and shots on target for / against)
  venue split    pts_venue5, gf_venue5, ga_venue5    (home team's last home games, away team's last away games)
  schedule       rest_days (days since last league match, max 14), n_prev (matches of history)
  strength       home_elo_pre, away_elo_pre, elo_diff
  league         div_E0, div_SP1, ...
"""
import numpy as np
import pandas as pd

from common import CLEAN_DIR, load_matches

ID_COLS = ["Div", "Date", "Season", "HomeTeam", "AwayTeam"]
SHORT_WINDOW, LONG_WINDOW = 5, 10


def team_match_table(df):
    """Two rows per match (one per team) so each team has a single timeline."""
    home = pd.DataFrame({
        "mid": df.index, "team": df["HomeTeam"], "venue": "H", "Date": df["Date"],
        "gf": df["FTHG"], "ga": df["FTAG"],
        "sf": df["HS"], "sa": df["AS"], "sotf": df["HST"], "sota": df["AST"],
    })
    away = pd.DataFrame({
        "mid": df.index, "team": df["AwayTeam"], "venue": "A", "Date": df["Date"],
        "gf": df["FTAG"], "ga": df["FTHG"],
        "sf": df["AS"], "sa": df["HS"], "sotf": df["AST"], "sota": df["HST"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "Date", "mid"]).reset_index(drop=True)
    long["pts"] = np.select([long["gf"] > long["ga"], long["gf"] == long["ga"]], [3, 1], default=0)
    return long


def rolling_mean(long, col, window, min_periods, by):
    """Mean of the previous `window` matches (shift(1) means the current match is excluded)."""
    return long.groupby(by)[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_periods).mean())


def team_features(long):
    out = pd.DataFrame({"mid": long["mid"], "venue": long["venue"]})
    for col in ["pts", "gf", "ga"]:
        out[f"{col}{SHORT_WINDOW}"] = rolling_mean(long, col, SHORT_WINDOW, 3, "team")
        out[f"{col}{LONG_WINDOW}"] = rolling_mean(long, col, LONG_WINDOW, 5, "team")
        out[f"{col}_venue5"] = rolling_mean(long, col, 5, 2, ["team", "venue"])
    for col in ["sf", "sa", "sotf", "sota"]:
        out[f"{col}5"] = rolling_mean(long, col, 5, 3, "team")
    out["rest_days"] = long.groupby("team")["Date"].diff().dt.days.clip(upper=14)
    out["n_prev"] = long.groupby("team").cumcount().clip(upper=20)
    return out


def build_features(df):
    f = team_features(team_match_table(df))
    home = f[f["venue"] == "H"].drop(columns="venue").set_index("mid").add_suffix("_h")
    away = f[f["venue"] == "A"].drop(columns="venue").set_index("mid").add_suffix("_a")
    feats = home.join(away)

    for c in ["pts5", "gf5", "ga5", "pts10", "sotf5", "sota5", "rest_days"]:
        feats[f"{c}_diff"] = feats[f"{c}_h"] - feats[f"{c}_a"]

    out = df[ID_COLS].join(feats)
    return out


def add_elo(out):
    path = CLEAN_DIR / "elo_features.csv"
    if not path.exists():
        raise SystemExit("elo_features.csv not found. Run elo_baseline.py first.")
    elo = pd.read_csv(path, parse_dates=["Date"])
    keys = ["Date", "HomeTeam", "AwayTeam"]
    merged = out.merge(elo[keys + ["home_elo_pre", "away_elo_pre", "elo_diff"]], on=keys, how="left")
    if merged["elo_diff"].isna().any():
        raise SystemExit("Some matches have no Elo values. Re-run clean.py and elo_baseline.py.")
    return merged


def leakage_check(df, out, n_checks=300, seed=0):
    """Recompute 'goals scored in the last 5 matches' by hand, using only earlier matches."""
    rng = np.random.default_rng(seed)
    checked = 0
    for i in rng.choice(len(df), size=min(n_checks, len(df)), replace=False):
        row = df.loc[i]
        for team, side, gf_col, ga_col in [(row["HomeTeam"], "h", "FTHG", "FTAG"),
                                           (row["AwayTeam"], "a", "FTAG", "FTHG")]:
            prior = df[(df["Date"] < row["Date"]) &
                       ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))].tail(SHORT_WINDOW)
            goals_for = np.where(prior["HomeTeam"] == team, prior["FTHG"], prior["FTAG"])
            expected = goals_for.mean() if len(prior) >= 3 else np.nan
            actual = out.loc[i, f"gf5_{side}"]
            same = (np.isnan(expected) and np.isnan(actual)) or np.isclose(expected, actual)
            assert same, (f"LEAK/MISMATCH at row {i} ({team}, {row['Date'].date()}): "
                          f"expected {expected}, got {actual}")
            checked += 1
    print(f"Leakage check passed ({checked} hand-checked values match).")


def main():
    df = load_matches()
    out = build_features(df)
    leakage_check(df, out)
    out = add_elo(out)
    out = pd.concat([out, pd.get_dummies(out["Div"], prefix="div", dtype=int)], axis=1)
    out["y"] = df["y"].to_numpy()

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_DIR / "features.csv", index=False)

    feature_cols = [c for c in out.columns if c not in ID_COLS + ["y"]]
    print(f"\nSaved {CLEAN_DIR / 'features.csv'}: {len(out)} matches, {len(feature_cols)} features")
    print("\nMissing values per feature (%), only features with any:")
    missing = (out[feature_cols].isna().mean() * 100).round(1)
    print(missing[missing > 0].sort_values(ascending=False).to_string())
    print("\nMissing values are expected early in the data (first matches of 2018/19) and for "
          "newly promoted teams; the models handle them.")


if __name__ == "__main__":
    main()
