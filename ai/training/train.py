"""ScoreLine feature engineering (stage 7b) and model/evaluate (stage 8)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

LABELS = ["A", "D", "H"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 7b: placeholder features. Must only use info known before kickoff."""
    # TODO: rolling form (last-N points/goals), Elo, head-to-head, rest days.
    return pd.DataFrame({"is_home": np.ones(len(df))}, index=df.index)


def train_model(X: pd.DataFrame, y: pd.Series):
    """Stage 8: fit a simple multinomial logistic regression baseline."""
    # TODO: Mixture-of-Experts — e.g. per-league/context experts with a gating model.
    return LogisticRegression(max_iter=1000).fit(X, y)


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Stage 8: log_loss + accuracy, compared against an always-home baseline."""
    proba = model.predict_proba(X)
    return {
        "log_loss": float(log_loss(y, proba, labels=model.classes_)),
        "accuracy": float(accuracy_score(y, model.predict(X))),
        "baseline_home_accuracy": float((y == "H").mean()),
    }
