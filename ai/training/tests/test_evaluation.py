import numpy as np
import pandas as pd
import pytest

from ai.training import evaluation as ev
from ai.training import preprocessing as pp
from ai.training import train as tr

Y = np.array(["H", "A", "D", "H"])
PERFECT = ev.one_hot(Y)
UNIFORM = np.full((4, 3), 1 / 3)


def test_log_loss_known_values():
    assert ev.per_row_log_loss(Y, PERFECT).max() == pytest.approx(0, abs=1e-9)
    assert ev.per_row_log_loss(Y, UNIFORM).mean() == pytest.approx(np.log(3))


def test_brier_known_values():
    assert ev.brier_multiclass(Y, PERFECT) == 0
    assert ev.brier_multiclass(Y, UNIFORM) == pytest.approx(2 / 3)  # (2/3)^2 + 2*(1/3)^2
    worst = ev.one_hot(np.array(["A", "H", "H", "A"]))
    assert ev.brier_multiclass(Y, worst) == 2


def test_calibration_table_and_ece():
    table = ev.calibration_table(Y, UNIFORM, n_bins=5)
    h = table[table["outcome"] == "H"].iloc[0]
    assert (h["n"], h["mean_pred"], h["observed"]) == (4, pytest.approx(1 / 3), 0.5)
    # ECE: |1/3-1/4| (A) + |1/3-1/4| (D) + |1/3-1/2| (H), averaged
    assert ev.expected_calibration_error(Y, UNIFORM) == pytest.approx((1 / 12 + 1 / 12 + 1 / 6) / 3)
    assert ev.expected_calibration_error(Y, PERFECT) == 0


def test_bootstrap_diff_sign_and_ci():
    y = np.array(["H", "A", "D"] * 40)
    better = 0.6 * ev.one_hot(y) + 0.4 / 3
    res = ev.bootstrap_diff(y, better, np.full((120, 3), 1 / 3), n_boot=200, seed=1)
    assert res["diff"] < 0 and res["ci_high"] < 0 and res["significant"]
    same = ev.bootstrap_diff(y, UNIFORM[:1].repeat(120, 0), UNIFORM[:1].repeat(120, 0), n_boot=50)
    assert same["diff"] == 0 and not same["significant"]
    # deterministic with a fixed seed
    assert ev.bootstrap_diff(y, better, UNIFORM[:1].repeat(120, 0), n_boot=100, seed=3) == \
        ev.bootstrap_diff(y, better, UNIFORM[:1].repeat(120, 0), n_boot=100, seed=3)


def test_confusion_table():
    cm = ev.confusion_table(Y, np.array(["H", "H", "D", "A"]))
    assert cm.loc["actual_H", "pred_H"] == 1 and cm.loc["actual_H", "pred_A"] == 1
    assert cm.loc["actual_A", "pred_H"] == 1 and cm.to_numpy().sum() == 4


def test_markdown():
    md = ev.to_markdown(pd.DataFrame({"m": ["x"], "v": [0.12345]}))
    assert md.splitlines() == ["| m | v |", "|---|---|", "| x | 0.1235 |"]


def _prepare(matches):
    df = pp.label_result(pp.clean(pp.load_api_matches(matches)))
    return df.join(tr.build_features(df))


def test_report_smoke():
    rng = np.random.default_rng(0)
    teams = list("ABCDEF")
    matches = []
    for i, d in enumerate(pd.date_range("2023-08-01", periods=120, freq="5D")):
        h, a = rng.choice(teams, 2, replace=False)
        matches.append({"id": i, "home_team": h, "away_team": a, "date": d.isoformat(),
                        "status": "FINISHED", "home_score": int(rng.integers(0, 4)),
                        "away_score": int(rng.integers(0, 3))})
    md = ev.report(_prepare(matches), n_boot=50)
    assert "### Test-set comparison" in md and "dummy_prior" in md and "Confusion matrix" in md

