"""ScoreLine feature engineering (stage 7b), model selection and evaluate (stage 8)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss

LABELS = ["A", "D", "H"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 7b: placeholder features. Must only use info known before kickoff."""
    # TODO: rolling form (last-N points/goals), Elo, head-to-head, rest days.
    return pd.DataFrame({"is_home": np.ones(len(df))}, index=df.index)


def train_model(X: pd.DataFrame, y: pd.Series):
    """Stage 8: fit a simple multinomial logistic regression baseline."""
    return LogisticRegression(max_iter=1000).fit(X, y)


# Model selection candidates. None is assumed best; selection is by CV log loss.
# TODO: future candidate - Mixture-of-Experts (e.g. per-context experts + gating),
# added here only if it beats these on time-series CV.
CANDIDATES = {
    "dummy_prior": DummyClassifier(strategy="prior"),
    "logreg": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=0),
}


def _aligned_proba(model, X, labels):
    """predict_proba with columns aligned to `labels` (folds may miss a class)."""
    proba = np.full((len(X), len(labels)), 1e-15)
    raw = model.predict_proba(X)
    for j, cls in enumerate(model.classes_):
        proba[:, labels.index(cls)] = raw[:, j]
    return proba / proba.sum(axis=1, keepdims=True)


def select_model(X: pd.DataFrame, y: pd.Series, candidates: dict = CANDIDATES, n_splits: int = 3):
    """Model selection: TimeSeriesSplit CV on train data, lowest mean log loss wins.

    Returns (best_name, best_model refit on all of X/y, scores DataFrame).
    """
    labels = sorted(y.unique())
    rows = []
    for name, est in candidates.items():
        losses = []
        for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
            m = clone(est).fit(X.iloc[tr_idx], y.iloc[tr_idx])
            losses.append(log_loss(y.iloc[va_idx], _aligned_proba(m, X.iloc[va_idx], labels), labels=labels))
        rows.append({"model": name, "mean_log_loss": float(np.mean(losses))})
    scores = pd.DataFrame(rows).sort_values("mean_log_loss").reset_index(drop=True)
    best = scores.loc[0, "model"]
    return best, clone(candidates[best]).fit(X, y), scores


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Stage 8: log_loss + accuracy, compared against an always-home baseline."""
    proba = model.predict_proba(X)
    return {
        "log_loss": float(log_loss(y, proba, labels=model.classes_)),
        "accuracy": float(accuracy_score(y, model.predict(X))),
        "baseline_home_accuracy": float((y == "H").mean()),
    }
