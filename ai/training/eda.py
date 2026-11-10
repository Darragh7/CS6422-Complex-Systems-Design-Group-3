"""Stage 6: explore on TRAIN only (never on test or the full timeline).

Pearson correlation of the model feature set (`features.FEATURE_COLUMNS`).
Run these helpers on the training split after `time_split` so exploration
cannot leak test-set structure into feature or model choices.
"""
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


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
