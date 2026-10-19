"""Bare sklearn-only Mixture of Experts candidate (no new dependencies).

Contexts are fixed by a pre-match feature: "home stronger" (elo_diff > 0) vs not.
One expert per context is trained on that context's rows; a gating logistic
regression learns p(context | x) and the output is the gate-weighted mix of
expert probabilities. Deliberately simple: it is one candidate in model
selection, not assumed to be the best.
TODO: learned/soft contexts (e.g. EM), more experts (league, season stage).
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _scaled_logreg(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


class ContextMoE(ClassifierMixin, BaseEstimator):
    def __init__(self, context_feature="elo_diff", threshold=0.0, C=0.1, min_rows=30):
        self.context_feature = context_feature
        self.threshold = threshold
        self.C = C
        self.min_rows = min_rows

    def _context(self, X):
        return (X[self.context_feature].to_numpy() > self.threshold).astype(int)

    def _aligned(self, model, X):
        out = np.zeros((len(X), len(self.classes_)))
        out[:, np.searchsorted(self.classes_, model.classes_)] = model.predict_proba(X)
        return out

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        ctx = self._context(X)
        self.experts_ = []
        for c in (0, 1):
            mask = ctx == c
            # too few rows or classes in a context -> fall back to class prior
            if mask.sum() >= self.min_rows and len(np.unique(y[mask])) > 1:
                est = _scaled_logreg(self.C)
            else:
                est = DummyClassifier(strategy="prior")
                mask = np.ones(len(y), bool) if mask.sum() == 0 else mask
            self.experts_.append(est.fit(X[mask], y[mask]))
        if len(np.unique(ctx)) > 1:
            self.gate_ = _scaled_logreg(1.0).fit(X, ctx)
        else:
            self.gate_ = DummyClassifier(strategy="prior").fit(X, ctx)
        return self

    def predict_proba(self, X):
        gate = np.zeros((len(X), 2))
        gate[:, self.gate_.classes_] = self.gate_.predict_proba(X)
        proba = sum(gate[:, [c]] * self._aligned(e, X) for c, e in enumerate(self.experts_))
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
