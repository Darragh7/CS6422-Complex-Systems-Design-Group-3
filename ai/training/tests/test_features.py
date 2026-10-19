import pandas as pd
import pytest

from ai.training.features import (DEFAULT_REST_DAYS, FEATURE_COLUMNS, H2H_NEUTRAL_PPG, REST_DAYS_CAP,
                                  build_features)


def make_df(rows):
    df = pd.DataFrame(rows, columns=["home_team", "away_team", "date", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


ROWS = [
    ("A", "B", "2024-08-01", 2, 0),
    ("C", "D", "2024-08-02", 1, 1),
    ("B", "C", "2024-08-08", 0, 3),
    ("D", "A", "2024-08-10", 1, 2),
    ("A", "C", "2024-08-15", 0, 0),
    ("B", "A", "2024-08-20", 1, 1),
]


def test_columns_and_no_nans():
    feats = build_features(make_df(ROWS))
    assert list(feats.columns) == FEATURE_COLUMNS
    assert not feats.isna().any().any()


@pytest.mark.parametrize("k", range(len(ROWS)))
def test_no_leakage_from_own_score(k):
    base = build_features(make_df(ROWS))
    changed = [list(r) for r in ROWS]
    changed[k][3], changed[k][4] = 9, 7
    feats = build_features(make_df(changed))
    # match k and everything before it must be identical
    pd.testing.assert_frame_equal(base.iloc[: k + 1], feats.iloc[: k + 1])


def test_rest_days():
    feats = build_features(make_df(ROWS))
    assert feats.loc[0, "home_rest_days"] == DEFAULT_REST_DAYS  # A first match
    assert feats.loc[0, "home_no_history"] == 1
    assert feats.loc[3, "away_rest_days"] == 9  # A: 08-01 -> 08-10
    assert feats.loc[5, "home_rest_days"] == 12  # B: 08-08 -> 08-20


def test_rolling_form():
    feats = build_features(make_df(ROWS))
    # A before match 4: W (3) then away W at D (3) -> ppg 3, gf 2, ga 0.5
    assert feats.loc[4, ["home_ppg", "home_gf", "home_ga"]].tolist() == [3.0, 2.0, 0.5]
    assert feats.loc[0, "home_ppg"] == 0.0


def test_head_to_head_default_and_history():
    feats = build_features(make_df(ROWS))
    assert feats.loc[0, "h2h_home_ppg"] == H2H_NEUTRAL_PPG and feats.loc[0, "h2h_n"] == 0
    # B hosts A at match 5; previously B lost 2-0 to A -> B has 0 ppg in h2h
    assert feats.loc[5, "h2h_home_ppg"] == 0.0 and feats.loc[5, "h2h_n"] == 1


def test_rest_days_capped():
    rows = ROWS + [("A", "D", "2024-12-01", 1, 0)]
    feats = build_features(make_df(rows))
    assert feats.iloc[-1]["home_rest_days"] == REST_DAYS_CAP


def test_elo_prior_only_and_direction():
    feats = build_features(make_df(ROWS))
    assert feats.loc[0, "elo_diff"] == 0.0  # both teams start equal
    # A beat B in match 0, so when B hosts A in match 5, B is rated lower
    assert feats.loc[5, "elo_diff"] < 0
    # leakage: changing the final match's result must not change its own Elo diff
    changed = [list(r) for r in ROWS]
    changed[5][3], changed[5][4] = 5, 0
    assert build_features(make_df(changed)).loc[5, "elo_diff"] == feats.loc[5, "elo_diff"]
