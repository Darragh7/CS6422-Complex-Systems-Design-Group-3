import numpy as np
import pandas as pd

from ai.training.moe import ContextMoE


def test_context_moe_fits_and_predicts_valid_probas():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"elo_diff": rng.normal(0, 100, 200), "home_ppg": rng.uniform(0, 3, 200)})
    y = np.where(X["elo_diff"] > 50, "H", np.where(X["elo_diff"] < -50, "A", "D"))
    model = ContextMoE(min_rows=10).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 3)
    assert np.allclose(proba.sum(axis=1), 1)
    assert list(model.classes_) == ["A", "D", "H"]
    assert (model.predict(X) == y).mean() > 0.6


def test_context_moe_falls_back_when_one_context_is_empty():
    X = pd.DataFrame({"elo_diff": np.ones(40), "x": np.arange(40.0)})
    y = np.array(["H", "A"] * 20)
    proba = ContextMoE(min_rows=5).fit(X, y).predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1)
