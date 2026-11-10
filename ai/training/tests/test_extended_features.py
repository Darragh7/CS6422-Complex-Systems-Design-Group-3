import numpy as np
import pandas as pd
import pytest

from ai.training import train as T
from ai.training.features import (BASELINE, FEATURE_COLUMNS, FEATURE_GROUPS, FORM_COLUMNS, WEAK_COLUMNS,
                                  build_extended_features, build_features, dc_score_matrix, dixon_coles,
                                  feature_set, table_gap)
from ai.training.tests.test_features import ROWS, make_df

TEAMS = list("ABCDEF")


def _season_rows(seed=0, weeks=16):
    """Round-robin-ish synthetic data: 3 matches per Saturday, 2 kickoffs, A strongest."""
    rng = np.random.default_rng(seed)
    rows, start = [], pd.Timestamp("2024-08-10")
    for w in range(weeks):
        order = rng.permutation(TEAMS)
        for j in range(3):
            h, a = order[2 * j], order[2 * j + 1]
            kick = start + pd.Timedelta(weeks=w, hours=12 + 3 * (j > 0))  # 2 matches share a kickoff
            hg = rng.poisson(2.5 if h == "A" else 1.4)
            ag = rng.poisson(2.2 if a == "A" else 1.1)
            rows.append((h, a, kick.isoformat(), int(hg), int(ag)))
    return rows


ALL_EXTENDED = [c for cols in FEATURE_GROUPS.values() for c in cols]


def test_extended_shape_columns_no_nulls():
    df = make_df(_season_rows())
    out = build_extended_features(df)
    assert len(out) == len(df)
    assert list(out.columns[:13]) == FEATURE_COLUMNS
    assert set(ALL_EXTENDED) <= set(out.columns)
    assert not out.isna().any().any()
    assert np.allclose(out["exp_total_goals"], out["exp_home_goals"] + out["exp_away_goals"])
    assert (out["abs_elo_diff"] >= 0).all() and (out["elo_balance"].between(0, 0.25)).all()


@pytest.mark.parametrize("k", [0, 5, 20, 33, 44])
def test_extended_no_leakage_from_own_score(k):
    rows = _season_rows()
    base = build_extended_features(make_df(rows))
    changed = [list(r) for r in rows]
    changed[k][3], changed[k][4] = 9, 7
    new = build_extended_features(make_df(changed))
    # match k (and anything at or before its kickoff) must be identical
    pd.testing.assert_frame_equal(base.iloc[: k + 1], new.iloc[: k + 1])
    # ...but later rows do react, so the features are not just constant
    assert not base.iloc[k + 1:].equals(new.iloc[k + 1:])


def test_same_kickoff_matches_do_not_see_each_other():
    rows = _season_rows()
    df = make_df(rows)
    # rows 1 and 2 share a kickoff: changing row 1 must not move row 2's league/table/DC features
    changed = [list(r) for r in rows]
    changed[1][3], changed[1][4] = 9, 7
    a, b = build_extended_features(df), build_extended_features(make_df(changed))
    pd.testing.assert_series_equal(a.loc[2], b.loc[2])


def test_dc_probabilities_valid():
    df = make_df(_season_rows())
    dc = dixon_coles(df)
    s = dc["dc_p_home"] + dc["dc_p_draw"] + dc["dc_p_away"]
    assert np.allclose(s, 1.0)
    assert dc["dc_p_draw"].between(0, 1).all() and (dc["dc_exp_total"] > 0).all()
    # A is the strongest team: once DC is fitted, A at home is favoured
    late = df.index[df["date"] > df["date"].iloc[len(df) // 2]]
    a_home = [i for i in late if df.loc[i, "home_team"] == "A"]
    assert dc.loc[a_home, "dc_p_home"].mean() > dc.loc[late, "dc_p_home"].mean()


def test_dc_score_matrix_sums_to_one_and_tau():
    p = dc_score_matrix(1.4, 1.1, -0.1)
    assert p.shape == (8, 8) and np.isclose(p.sum(), 1.0) and (p >= 0).all()
    indep = dc_score_matrix(1.4, 1.1, 0.0)
    assert p[0, 0] > indep[0, 0] and p[1, 1] > indep[1, 1]  # rho < 0 inflates 0-0 and 1-1


def test_table_gap_tiny_season():
    rows = [
        ("A", "B", "2024-08-10T14:00:00Z", 2, 0),
        ("C", "D", "2024-08-10T14:00:00Z", 1, 0),
        ("A", "C", "2024-08-17T14:00:00Z", 0, 0),
        ("B", "D", "2024-08-17T16:00:00Z", 1, 1),
        ("B", "A", "2025-08-16T14:00:00Z", 1, 0),  # new season: table resets
    ]
    t = table_gap(make_df(rows))
    # matchweek 1: nobody has played -> gaps 0 and flagged
    assert t.loc[0].tolist() == [0.0, 0.0, 0.0, 1] and t.loc[1, "table_mw1"] == 1
    # after MW1: A 1st (3 pts, +2), C 2nd (3 pts, +1), D 3rd (0, -1), B 4th (0, -2)
    assert t.loc[2, "ppg_gap"] == 0.0 and t.loc[2, "pos_gap"] == 1.0 and t.loc[2, "table_mw1"] == 0
    # B hosts D at 16:00; the 14:00 A-C draw the same day IS in the table (earlier kickoff)
    # after A-C draw: A 4 pts, C 4 pts (A +2 vs C +1), D 0 (-1), B 0 (-2)
    assert t.loc[3, "ppg_gap"] == 0.0 and t.loc[3, "pos_gap"] == -1.0
    assert t.loc[4, "table_mw1"] == 1 and t.loc[4, "ppg_gap"] == 0.0
    assert (t["abs_ppg_gap"] == t["ppg_gap"].abs()).all()


def test_table_gap_ppg():
    rows = [("A", "B", "2024-08-10T14:00:00Z", 2, 0), ("C", "D", "2024-08-10T14:00:00Z", 1, 1),
            ("A", "D", "2024-08-17T14:00:00Z", 0, 1)]
    t = table_gap(make_df(rows))
    assert t.loc[2, "ppg_gap"] == 2.0 and t.loc[2, "abs_ppg_gap"] == 2.0  # A 3 ppg vs D 1 ppg


def test_feature_set_variants():
    lean = feature_set(drop=WEAK_COLUMNS)
    assert len(BASELINE) == 13 and len(lean) == 8 and not set(WEAK_COLUMNS) & set(lean)
    rep = feature_set("ewm_form", base=lean, replace_form=True)
    assert rep == ["h2h_home_ppg", "elo_diff", "ewm_gd_diff"] and not set(FORM_COLUMNS) & set(rep)
    assert feature_set("dixon_coles")[-3:] == FEATURE_GROUPS["dixon_coles"]


def test_eval_helpers():
    y = pd.Series(["H", "D", "A", "H"])
    p = pd.DataFrame([[0.2, 0.3, 0.5]] * 4, columns=T.LABELS)
    m = T.prob_metrics(y, p)
    assert m["n_draws_predicted"] == 0 and np.isclose(m["mean_p_draw"], 0.3)
    assert np.isclose(m["draw_outcome_log_loss"], -np.log(0.3))
    d, lo, hi = T.paired_bootstrap_ci(y, p, p, n_boot=200)
    assert d == 0 and lo == 0 and hi == 0


def test_existing_baseline_untouched():
    df = make_df(ROWS)
    pd.testing.assert_frame_equal(build_extended_features(df)[FEATURE_COLUMNS], build_features(df))
