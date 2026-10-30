"""Stage 8 evaluation & reporting: calibration and comparison against baselines.

Tables only (matplotlib is not in backend/requirements.txt). report() returns
markdown so results can be pasted into PRs/notebooks without a docs page.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix

from . import preprocessing as pp
from . import train as tr

LABELS = tr.LABELS  # ["A", "D", "H"]
EPS = 1e-15


# ---------- metric primitives ----------

def one_hot(y, labels=LABELS) -> np.ndarray:
    y = np.asarray(y)
    return (y[:, None] == np.asarray(labels)[None, :]).astype(float)


def per_row_log_loss(y, proba, labels=LABELS) -> np.ndarray:
    """-log p(true outcome) for each row (proba columns ordered as labels)."""
    p = np.clip(np.asarray(proba, float), EPS, 1)
    return -np.log((one_hot(y, labels) * p).sum(axis=1))


def brier_multiclass(y, proba, labels=LABELS) -> float:
    """Mean over rows of sum_k (p_k - 1[y=k])^2. 0 is perfect, 2 is worst."""
    return float(((np.asarray(proba, float) - one_hot(y, labels)) ** 2).sum(axis=1).mean())


def calibration_table(y, proba, labels=LABELS, n_bins: int = 5) -> pd.DataFrame:
    """Per outcome: equal-width bins of predicted prob vs observed frequency."""
    onehot, proba = one_hot(y, labels), np.asarray(proba, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for k, label in enumerate(labels):
        bins = np.clip(np.digitize(proba[:, k], edges[1:-1]), 0, n_bins - 1)
        for b in range(n_bins):
            mask = bins == b
            if mask.any():
                rows.append({"outcome": label, "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                             "n": int(mask.sum()), "mean_pred": float(proba[mask, k].mean()),
                             "observed": float(onehot[mask, k].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(y, proba, labels=LABELS, n_bins: int = 5) -> float:
    """Class-wise ECE: mean over outcomes of sum_bins (n_b/N) |mean_pred - observed|."""
    table = calibration_table(y, proba, labels, n_bins)
    n = len(np.asarray(y))
    gaps = (table["n"] / n) * (table["mean_pred"] - table["observed"]).abs()
    return float(gaps.groupby(table["outcome"]).sum().mean())


def bootstrap_diff(y, proba_model, proba_base, labels=LABELS, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired bootstrap of mean log-loss difference (model - base). Negative = model better."""
    d = per_row_log_loss(y, proba_model, labels) - per_row_log_loss(y, proba_base, labels)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(hi < 0 or lo > 0)}


def confusion_table(y, pred, labels=LABELS) -> pd.DataFrame:
    """Rows = actual, columns = predicted."""
    cm = confusion_matrix(y, pred, labels=labels)
    return pd.DataFrame(cm, index=[f"actual_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


# ---------- model comparison ----------

def _fit_proba(est, train_df, test_df, target="result"):
    model = clone(est).fit(train_df[tr.FEATURE_COLUMNS], train_df[target])
    X = test_df[tr.FEATURE_COLUMNS]
    return model, tr._aligned_proba(model, X, LABELS)


def compare_candidates(train_df, test_df, candidates=None, target="result",
                       baseline="dummy_prior", n_boot=1000, seed=0):
    """Fit every candidate on train, score on test; bootstrap CI vs the baseline."""
    candidates = candidates or tr.CANDIDATES
    y = test_df[target].to_numpy()
    probas = {name: _fit_proba(est, train_df, test_df, target)[1] for name, est in candidates.items()}
    rows = []
    for name, p in probas.items():
        boot = bootstrap_diff(y, p, probas[baseline], n_boot=n_boot, seed=seed)
        rows.append({"model": name, "log_loss": float(per_row_log_loss(y, p).mean()),
                     "accuracy": float(accuracy_score(y, np.asarray(LABELS)[p.argmax(axis=1)])),
                     "brier": brier_multiclass(y, p), "ece": expected_calibration_error(y, p),
                     "ll_diff_vs_prior": boot["diff"], "ci_low": boot["ci_low"],
                     "ci_high": boot["ci_high"], "significant": boot["significant"]})
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True), probas


# ---------- report ----------

def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Minimal markdown table (avoids the optional tabulate dependency)."""
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def report(df: pd.DataFrame, test_frac: float = 0.2, selected: str | None = None,
           n_boot: int = 1000, seed: int = 0) -> str:
    """Markdown results report for a labelled + featured df (stages 5 and 7 applied)."""
    train_df, test_df = pp.time_split(df, test_frac)
    y = test_df["result"].to_numpy()
    table, probas = compare_candidates(train_df, test_df, n_boot=n_boot, seed=seed)
    selected = selected or tr.select_model(train_df[tr.FEATURE_COLUMNS], train_df["result"])[0]
    p_sel = probas[selected]
    parts = [
        f"## ScoreLine evaluation ({len(train_df)} train / {len(test_df)} test matches, "
        f"test {test_df['date'].min():%Y-%m-%d} to {test_df['date'].max():%Y-%m-%d})",
        f"Always-home accuracy: {(y == 'H').mean():.4f}. Selected by CV: `{selected}`. "
        f"CI = paired bootstrap 95% ({n_boot} resamples, seed {seed}) of test log-loss "
        "difference vs `dummy_prior` (negative = better).",
        "### Test-set comparison", to_markdown(table),
        f"### Confusion matrix (`{selected}`)",
        to_markdown(confusion_table(y, np.asarray(LABELS)[p_sel.argmax(axis=1)]).reset_index(names="")),
        f"### Calibration (`{selected}`, ECE {expected_calibration_error(y, p_sel):.4f})",
        to_markdown(calibration_table(y, p_sel)),
    ]
    return "\n\n".join(parts)
