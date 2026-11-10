"""Stage 6: explore on TRAIN only (never on test or the full timeline).

Pearson correlation of the model feature set (`features.FEATURE_COLUMNS`).
Run these helpers on the training split after `time_split` so exploration
cannot leak test-set structure into feature or model choices.
"""
import numpy as np
import pandas as pd

from .features import ELO_HOME_ADV, FEATURE_COLUMNS, _team_long

RESULTS = ["H", "D", "A"]
EWM_SPAN = 10


def feature_correlation(features_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Pairwise correlation of FEATURE_COLUMNS. Use the train split only."""
    return features_df[list(FEATURE_COLUMNS)].corr(method=method)


def top_correlated_pairs(corr: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Upper-triangle pairs with |r| >= threshold, sorted by |r| descending."""
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = float(corr.loc[a, b])
            if np.isfinite(r) and abs(r) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "r": r, "abs_r": abs(r)})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "r", "abs_r"]).sort_values(
        "abs_r", ascending=False
    ).reset_index(drop=True)


def target_correlation(features_df: pd.DataFrame, labels, goal_diff=None) -> pd.DataFrame:
    """Pearson r of each feature with numeric encodings of the match result.

    `labels` is the H/D/A series (aligned to `features_df`). Encoded as a
    home-win indicator and a draw indicator. If `goal_diff` is given, or
    `home_score`/`away_score` are present on `features_df`, also correlate
    with goal difference (home minus away).
    """
    labels = pd.Series(labels, index=getattr(labels, "index", features_df.index))
    targets = {
        "home_win": (labels == "H").astype(float),
        "draw": (labels == "D").astype(float),
    }
    if goal_diff is None and {"home_score", "away_score"}.issubset(features_df.columns):
        goal_diff = features_df["home_score"] - features_df["away_score"]
    if goal_diff is not None:
        targets["goal_diff"] = pd.Series(goal_diff, index=getattr(goal_diff, "index", features_df.index))
    X = features_df[list(FEATURE_COLUMNS)]
    rows = {}
    for name, target in targets.items():
        rows[name] = X.corrwith(target.reindex(X.index), method="pearson")
    return pd.DataFrame(rows, index=FEATURE_COLUMNS)



def outcome_rates(values: pd.Series, labels: pd.Series, bins) -> pd.DataFrame:
    """Empirical P(H/D/A) and match count per bin of `values` (train only).

    `bins` is an int (equal-count quantile bins) or a list of edges.
    """
    values = pd.Series(values, index=labels.index)
    cut = pd.qcut(values, bins, duplicates="drop") if isinstance(bins, int) else pd.cut(values, bins)
    rates = pd.crosstab(cut, labels, normalize="index").reindex(columns=RESULTS, fill_value=0.0)
    rates["n"] = cut.value_counts().reindex(rates.index)
    return rates


def vif(features_df: pd.DataFrame, cols=None) -> pd.Series:
    """Variance inflation factor per column: diag of the inverse correlation matrix."""
    cols = list(cols or FEATURE_COLUMNS)
    corr = features_df[cols].corr().to_numpy()
    return pd.Series(np.diag(np.linalg.pinv(corr)), index=cols, name="vif").sort_values(ascending=False)


def candidate_features(df: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """PROPOSED features for EDA only (not in FEATURE_COLUMNS, not used by train.py).

    Prior-only like features.py: every per-team stat is shift(1)-ed before ewm,
    league averages are shifted expanding means. `base` = build_features(df).
    """
    long = _team_long(df)
    long["gd"] = long["gf"] - long["ga"]
    g = long.groupby("team", sort=False)
    for col in ("gf", "ga", "gd"):
        long[f"ewm_{col}"] = g[col].transform(lambda s: s.shift(1).ewm(span=EWM_SPAN, min_periods=1).mean())
    # venue-specific form: points in previous 5 home (or away) games only
    long["venue_ppg"] = long.groupby(["team", "side"], sort=False)["pts"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    h = long[long["side"] == "home"].set_index("match").reindex(df.index)
    a = long[long["side"] == "away"].set_index("match").reindex(df.index)

    out = pd.DataFrame(index=df.index)
    out["abs_elo_diff"] = base["elo_diff"].abs()
    out["elo_exp_home"] = 1 / (1 + 10 ** (-(base["elo_diff"] + ELO_HOME_ADV) / 400))
    out["ewm_gd_diff"] = (h["ewm_gd"] - a["ewm_gd"]).fillna(0.0)
    out["home_venue_ppg"] = h["venue_ppg"].fillna(1.5)
    out["away_venue_ppg"] = a["venue_ppg"].fillna(1.5)
    # Poisson-style: league prior average x attack strength x opponent defence weakness
    ordered = df.sort_values("date", kind="stable")
    lg_home = ordered["home_score"].shift(1).expanding().mean().reindex(df.index).fillna(1.5)
    lg_away = ordered["away_score"].shift(1).expanding().mean().reindex(df.index).fillna(1.2)
    lg_team = (lg_home + lg_away) / 2
    out["exp_home_goals"] = (lg_home * (h["ewm_gf"] / lg_team) * (a["ewm_ga"] / lg_team)).fillna(lg_home)
    out["exp_away_goals"] = (lg_away * (a["ewm_gf"] / lg_team) * (h["ewm_ga"] / lg_team)).fillna(lg_away)
    out["exp_total_goals"] = out["exp_home_goals"] + out["exp_away_goals"]
    return out
