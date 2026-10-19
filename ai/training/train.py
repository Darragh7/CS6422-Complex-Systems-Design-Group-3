"""ScoreLine feature engineering (stage 7b), model selection and evaluate (stage 8)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, build_features  # noqa: F401 (re-exported)
from .moe import ContextMoE

LABELS = ["A", "D", "H"]


def train_model(X: pd.DataFrame, y: pd.Series):
    """Stage 8: fit the scaled logistic regression candidate directly."""
    return clone(CANDIDATES["logreg"]).fit(X, y)


# Model selection candidates. None is assumed best; selection is by CV log loss.
CANDIDATES = {
    "dummy_prior": DummyClassifier(strategy="prior"),
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=5, min_samples_leaf=20, random_state=0
    ),
    "hist_gb": HistGradientBoostingClassifier(
        learning_rate=0.03, max_depth=3, max_iter=150, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    ),
    # Bare sklearn-only Mixture of Experts (see moe.py); just another candidate.
    "moe_context": ContextMoE(),
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


def evaluate(model, X: pd.DataFrame, y: pd.Series, y_train: pd.Series | None = None) -> dict:
    """Stage 8: log_loss + accuracy vs always-home and (if y_train given) class-prior baselines."""
    labels = sorted(set(y) | set(model.classes_))
    result = {
        "log_loss": float(log_loss(y, _aligned_proba(model, X, labels), labels=labels)),
        "accuracy": float(accuracy_score(y, model.predict(X))),
        "baseline_home_accuracy": float((y == "H").mean()),
    }
    if y_train is not None:
        prior = DummyClassifier(strategy="prior").fit(np.zeros((len(y_train), 1)), y_train)
        Z = np.zeros((len(y), 1))
        result["baseline_prior_log_loss"] = float(log_loss(y, _aligned_proba(prior, Z, labels), labels=labels))
        result["baseline_prior_accuracy"] = float(accuracy_score(y, prior.predict(Z)))
    return result
