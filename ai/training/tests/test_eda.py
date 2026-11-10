import numpy as np
import pandas as pd
import pytest

from ai.training.eda import feature_correlation, target_correlation, top_correlated_pairs
from ai.training.features import FEATURE_COLUMNS


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(size=n) for c in FEATURE_COLUMNS}
    # Plant a near-perfect pair and a moderate pair.
    data["away_ppg"] = data["home_ppg"] + rng.normal(scale=0.01, size=n)
    data["away_gf"] = 0.6 * data["home_gf"] + rng.normal(scale=0.8, size=n)
    return pd.DataFrame(data)


def test_correlation_symmetric_unit_diagonal():
    corr = feature_correlation(_synthetic())
    assert list(corr.columns) == list(FEATURE_COLUMNS)
    assert list(corr.index) == list(FEATURE_COLUMNS)
    assert np.allclose(corr.to_numpy(), corr.T.to_numpy())
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)


def test_top_pairs_detects_threshold_on_synthetic():
    corr = feature_correlation(_synthetic())
    pairs = top_correlated_pairs(corr, threshold=0.9)
    assert list(pairs.columns) == ["feature_a", "feature_b", "r", "abs_r"]
    found = {frozenset((a, b)) for a, b in zip(pairs["feature_a"], pairs["feature_b"])}
    assert frozenset(("home_ppg", "away_ppg")) in found
    assert (pairs["abs_r"] >= 0.9).all()
    assert (pairs["feature_a"] != pairs["feature_b"]).all()
    assert pairs["abs_r"].is_monotonic_decreasing
    # The planted moderate pair should appear at 0.3 but not at 0.9.
    lo = top_correlated_pairs(corr, threshold=0.3)
    lo_found = {frozenset((a, b)) for a, b in zip(lo["feature_a"], lo["feature_b"])}
    assert frozenset(("home_gf", "away_gf")) in lo_found
    assert frozenset(("home_gf", "away_gf")) not in found


def test_top_pairs_empty_when_nothing_clears_threshold():
    corr = feature_correlation(_synthetic())
    pairs = top_correlated_pairs(corr, threshold=1.01)
    assert list(pairs.columns) == ["feature_a", "feature_b", "r", "abs_r"]
    assert pairs.empty


def test_target_correlation_home_win_sign():
    n = 80
    rng = np.random.default_rng(1)
    X = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLUMNS})
    X["elo_diff"] = np.linspace(-2, 2, n)
    # Higher elo_diff -> more home wins; labels otherwise mixed.
    labels = pd.Series(np.where(X["elo_diff"] > 0, "H", "A"))
    labels.iloc[::7] = "D"
    out = target_correlation(X, labels, goal_diff=X["elo_diff"])
    assert list(out.index) == list(FEATURE_COLUMNS)
    assert set(out.columns) == {"home_win", "draw", "goal_diff"}
    assert out.loc["elo_diff", "home_win"] > 0.7
    assert out.loc["elo_diff", "goal_diff"] == pytest.approx(1.0)


# --- outcome_rates / vif / proposed candidate features (EDA only, not wired into training) ---
from ai.training.eda import candidate_features, outcome_rates, vif  # noqa: E402
from ai.training.features import build_features  # noqa: E402
from ai.training.tests.test_features import ROWS, make_df  # noqa: E402


def test_outcome_rates_rows_sum_to_one():
    labels = pd.Series(["H", "D", "A", "H"] * 10)
    out = outcome_rates(pd.Series(np.arange(40.0)), labels, 4)
    assert list(out.columns) == ["H", "D", "A", "n"]
    assert np.allclose(out[["H", "D", "A"]].sum(axis=1), 1.0)
    assert out["n"].sum() == 40


def test_vif_flags_collinear_pair():
    X = _synthetic()
    v = vif(X)
    assert v["home_ppg"] > 100 and v["away_ppg"] > 100  # planted near-duplicate
    assert v["elo_diff"] < 2


def _cand(rows):
    df = make_df(rows)
    return candidate_features(df, build_features(df))


def test_candidate_shape_and_no_nans():
    out = _cand(ROWS)
    assert len(out) == len(ROWS)
    assert not out.isna().any().any()
    assert (out["abs_elo_diff"] >= 0).all()
    assert np.allclose(out["exp_total_goals"], out["exp_home_goals"] + out["exp_away_goals"])


@pytest.mark.parametrize("k", range(len(ROWS)))
def test_candidate_no_leakage_from_own_score(k):
    base = _cand(ROWS)
    changed = [list(r) for r in ROWS]
    changed[k][3], changed[k][4] = 9, 7
    pd.testing.assert_frame_equal(base.iloc[: k + 1], _cand(changed).iloc[: k + 1])


def test_candidate_venue_form_uses_same_venue_only():
    out = _cand(ROWS)
    # A hosts C at match 4; A's only previous HOME game was a win (match 0) -> 3.0,
    # shrunk to the 1.5 prior because < 3 venue games: (3 + 2 * 1.5) / 3 = 2.0
    assert out.loc[4, "home_venue_ppg"] == 2.0
    # B hosts A at match 5; A's only previous AWAY game was a win at D (match 3) -> same
    assert out.loc[5, "away_venue_ppg"] == 2.0
