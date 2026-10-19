"""Stage 7b: match features computed strictly from PRIOR matches (no leakage).

Every per-team statistic is shift(1)-ed before rolling, so match k only sees
matches that kicked off before it. Build features on the full chronological
dataset *before* time_split so test matches can use training-period history.
"""
import numpy as np
import pandas as pd

WINDOW = 5
DEFAULT_REST_DAYS = 7.0  # used when a team has no previous match in the data
H2H_NEUTRAL_PPG = 1.5  # midpoint of 0..3 points, used when teams haven't met

TEAM_STATS = ["ppg", "gf", "ga", "rest_days"]
FEATURE_COLUMNS = (
    [f"{side}_{s}" for side in ("home", "away") for s in TEAM_STATS]
    + ["home_no_history", "away_no_history", "h2h_home_ppg", "h2h_n"]
)


def _points(gf: pd.Series, ga: pd.Series) -> pd.Series:
    return np.select([gf > ga, gf == ga], [3, 1], 0).astype(float)


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) with that team's goals for/against and points."""
    home = pd.DataFrame({"match": df.index, "date": df["date"], "team": df["home_team"],
                         "side": "home", "gf": df["home_score"], "ga": df["away_score"]})
    away = pd.DataFrame({"match": df.index, "date": df["date"], "team": df["away_team"],
                         "side": "away", "gf": df["away_score"], "ga": df["home_score"]})
    long = pd.concat([home, away], ignore_index=True).sort_values(["date", "match"], kind="stable")
    long["pts"] = _points(long["gf"], long["ga"])
    return long


def team_form(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Per (match, side): rolling ppg/gf/ga over previous `window` matches + rest days."""
    long = _team_long(df)
    g = long.groupby("team", sort=False)
    for col, out in (("pts", "ppg"), ("gf", "gf"), ("ga", "ga")):
        # shift(1) BEFORE rolling: exclude the current match.
        long[out] = g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["rest_days"] = g["date"].diff().dt.total_seconds() / 86400
    long["no_history"] = long["ppg"].isna().astype(int)
    return long


def head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Home team's points per game in previous meetings (either venue) + count."""
    ordered = df.sort_values("date", kind="stable")
    history: dict[frozenset, list[tuple[str, float]]] = {}
    ppg, n = {}, {}
    for idx, row in ordered.iterrows():
        key = frozenset((row["home_team"], row["away_team"]))
        past = history.setdefault(key, [])
        mine = [p for team, p in past if team == row["home_team"]]
        n[idx] = len(mine)
        ppg[idx] = float(np.mean(mine)) if mine else H2H_NEUTRAL_PPG
        # record AFTER computing features for this match
        hp = _points(pd.Series([row["home_score"]]), pd.Series([row["away_score"]]))[0]
        past += [(row["home_team"], hp), (row["away_team"], {3: 0.0, 1: 1.0, 0: 3.0}[int(hp)])]
    return pd.DataFrame({"h2h_home_ppg": pd.Series(ppg), "h2h_n": pd.Series(n)})


def build_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Feature table indexed like df. Expects clean() output (finished, parsed dates).

    Early-season gaps: rolling stats -> 0 with *_no_history=1; rest days -> 7.
    """
    # TODO: when load_static_csv is implemented, concatenate the static history
    # with API matches (dedupe by id) BEFORE calling this, so form/h2h use it.
    long = team_form(df, window)
    out = pd.DataFrame(index=df.index)
    for side in ("home", "away"):
        part = long[long["side"] == side].set_index("match")
        for s in TEAM_STATS:
            out[f"{side}_{s}"] = part[s]
        out[f"{side}_no_history"] = part["no_history"]
    for side in ("home", "away"):
        out[[f"{side}_ppg", f"{side}_gf", f"{side}_ga"]] = out[[f"{side}_ppg", f"{side}_gf", f"{side}_ga"]].fillna(0.0)
        out[f"{side}_rest_days"] = out[f"{side}_rest_days"].fillna(DEFAULT_REST_DAYS)
    out = out.join(head_to_head(df))
    return out[FEATURE_COLUMNS]
