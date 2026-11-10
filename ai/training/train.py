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


# --- ablation / evaluation helpers (train-only CV, then a single test comparison) ---

def oof_proba(est, X: pd.DataFrame, y: pd.Series, n_splits: int = 3, labels=LABELS):
    """Out-of-fold probabilities on the TimeSeriesSplit validation rows (same folds as select_model)."""
    idx, probs = [], []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = clone(est).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        idx.append(va_idx)
        probs.append(_aligned_proba(m, X.iloc[va_idx], labels))
    idx = np.concatenate(idx)
    return y.iloc[idx], pd.DataFrame(np.vstack(probs), index=y.index[idx], columns=labels)


def prob_metrics(y: pd.Series, proba: pd.DataFrame) -> dict:
    """Log loss, accuracy, multiclass Brier, draw metrics for an (n, A/D/H) probability frame."""
    labels = list(proba.columns)
    P = np.clip(proba.to_numpy(), 1e-15, 1)
    onehot = (np.asarray(y)[:, None] == np.array(labels)[None, :]).astype(float)
    pd_ = P[:, labels.index("D")]
    is_d = np.asarray(y) == "D"
    pred = np.array(labels)[P.argmax(axis=1)]
    return {
        "log_loss": float(log_loss(y, P, labels=labels)),
        "accuracy": float((pred == np.asarray(y)).mean()),
        "brier": float(((P - onehot) ** 2).sum(axis=1).mean()),
        "draw_outcome_log_loss": float(-np.log(pd_[is_d]).mean()),  # mean -log P(D) on actual draws
        "draw_binary_log_loss": float(log_loss(is_d, pd_, labels=[False, True])),
        "mean_p_draw": float(pd_.mean()),
        "draw_rate": float(is_d.mean()),
        "n_draws_predicted": int((pred == "D").sum()),
    }


def paired_bootstrap_ci(y, proba_a: pd.DataFrame, proba_b: pd.DataFrame, n_boot: int = 10000, seed: int = 0):
    """95% CI of mean per-match log loss difference (a - b), resampling matches with replacement."""
    labels = list(proba_a.columns)
    col = np.searchsorted(labels, np.asarray(y))
    rows = np.arange(len(y))
    diff = (-np.log(np.clip(proba_a.to_numpy()[rows, col], 1e-15, 1))
            + np.log(np.clip(proba_b.to_numpy()[rows, col], 1e-15, 1)))
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def cv_ablation(X_all: pd.DataFrame, y: pd.Series, variants: dict, n_splits: int = 3,
                candidates: dict = CANDIDATES) -> pd.DataFrame:
    """select_model CV log loss per model for each variant (name -> column list). Train data only."""
    rows = []
    for name, cols in variants.items():
        best, _, scores = select_model(X_all[cols], y, candidates, n_splits)
        r = {"variant": name, "n_features": len(cols), "best_model": best,
             "best_cv_log_loss": float(scores.loc[0, "mean_log_loss"])}
        r.update(dict(zip(scores["model"], scores["mean_log_loss"])))
        rows.append(r)
    return pd.DataFrame(rows)
