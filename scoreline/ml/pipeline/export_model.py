"""ML-11: export a versioned model artifact for the prediction pipeline.

Usage (from ml/pipeline):
    python export_model.py                      # version defaults to today's date
    python export_model.py --version v1

Run this AFTER final_test.py. It trains the production models on ALL matches with known
results (including the validation, test and current seasons) and saves:

    models/scoreline_<version>/
        lr_pipeline.joblib      logistic regression (with imputer + scaler)
        boosting_model.joblib   XGBoost (or scikit-learn gradient boosting)
        metadata.json           version, training date, data range, features, settings, metrics
        README.txt              how to load it, and a security note

The final ensemble is the plain average of four sets of probabilities:
Elo, Poisson, logistic regression and boosting. Elo and Poisson are recomputed from the
match history at prediction time (they have no model file), and their settings are stored
in metadata.json so the prediction job uses exactly the same ones.

Settings (C and number of trees) are chosen on the last full season (the test season) with
the earlier seasons as training data; then both models are re-fitted on everything.
"""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

import elo_baseline as eb
import poisson_baseline as pb
import train_models as tm
from common import CLEAN_DIR, RESULTS_DIR, TEST_SEASON, VAL_SEASON, load_matches, log_loss


def read_metrics(filename):
    path = RESULTS_DIR / filename
    if not path.exists():
        return None
    table = pd.read_csv(path)
    row = table[table["model"].str.startswith("Ensemble")]
    if row.empty:
        return None
    return row.iloc[0][["log_loss", "brier", "rps", "accuracy"]].to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=datetime.now().strftime("%Y%m%d"))
    args = parser.parse_args()

    matches = load_matches()
    feats = pd.read_csv(CLEAN_DIR / "features.csv", parse_dates=["Date"])
    if len(matches) != len(feats):
        raise SystemExit("features.csv does not match matches.csv. Re-run clean.py and features.py.")

    features = [c for c in feats.columns if c not in tm.ID_COLS + ["y"]]
    X, y = feats[features], feats["y"].astype(int)

    # Choose settings with the last full season as the mini validation set
    inner_val = (matches["Season"] == TEST_SEASON).to_numpy()
    inner_tr = (matches["Season"] <= VAL_SEASON).to_numpy()
    best_c, best_ll = None, np.inf
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        m = tm.make_lr(C).fit(X[inner_tr], y[inner_tr])
        ll = log_loss(m.predict_proba(X[inner_val]), y[inner_val].to_numpy())
        if ll < best_ll:
            best_c, best_ll = C, ll
    lr = tm.make_lr(best_c).fit(X, y)

    if tm.XGBClassifier is not None:
        probe = tm.XGBClassifier(n_estimators=1000, early_stopping_rounds=50,
                                 eval_metric="mlogloss", **tm.xgb_params())
        probe.fit(X[inner_tr], y[inner_tr], eval_set=[(X[inner_val], y[inner_val])], verbose=False)
        n_trees = int(probe.best_iteration) + 1
        boost = tm.XGBClassifier(n_estimators=n_trees, eval_metric="mlogloss", **tm.xgb_params()).fit(X, y)
        boost_type = "XGBoost"
    else:
        n_trees = 300
        boost = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=n_trees, max_depth=3,
                                               l2_regularization=5.0, random_state=0).fit(X, y)
        boost_type = "scikit-learn HistGradientBoosting"

    # Elo probability settings, fitted on all matches (ratings use only past matches)
    home_pre, away_pre, _ = eb.run_elo(matches)
    diff = home_pre + eb.HOME_ADV - away_pre
    scale, cut, _ = eb.fit_probability_model(diff, matches["y"].to_numpy().astype(int))

    # Save
    name = f"scoreline_{args.version}"
    out_dir = Path("models") / name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, out_dir / "lr_pipeline.joblib")
    joblib.dump(boost, out_dir / "boosting_model.joblib")

    metadata = {
        "model_version": name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {"first_match": str(matches["Date"].min().date()),
                 "last_match": str(matches["Date"].max().date()),
                 "n_matches": int(len(matches))},
        "ensemble": {"members": ["Elo", "Poisson", "Logistic regression", boost_type],
                     "combination": "simple average of probabilities; column order [home, draw, away]"},
        "features": features,
        "logistic_regression": {"C": best_c},
        "boosting": {"type": boost_type, "n_trees": n_trees},
        "elo": {"K": eb.K, "HOME_ADV": eb.HOME_ADV, "MEAN": eb.MEAN, "NEW_TEAM": eb.NEW_TEAM,
                "REGRESS": eb.REGRESS, "probability_scale": float(scale), "draw_width": float(cut)},
        "poisson": {"WINDOW_DAYS": pb.WINDOW_DAYS, "HALF_LIFE_DAYS": pb.HALF_LIFE_DAYS,
                    "ALPHA": pb.ALPHA, "MAX_GOALS": pb.MAX_GOALS},
        "metrics_validation_2024_25": read_metrics("evaluation_val.csv"),
        "metrics_final_test_2025_26": read_metrics("final_test.csv"),
        "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "README.txt").write_text(
        f"{name}\n\nLoad with:\n    import joblib\n"
        f"    lr = joblib.load('lr_pipeline.joblib')\n    boost = joblib.load('boosting_model.joblib')\n\n"
        "Inputs must have the exact feature columns listed in metadata.json, in that order.\n"
        "Final probabilities = average of Elo, Poisson, logistic regression and boosting.\n\n"
        "SECURITY: .joblib files can run code when loaded. Only load files that the team produced\n"
        "and stored in this repository or a trusted location. Never load one from an unknown source.\n",
        encoding="utf-8")

    # Loading test: reload from disk and check the predictions are identical
    lr2 = joblib.load(out_dir / "lr_pipeline.joblib")
    boost2 = joblib.load(out_dir / "boosting_model.joblib")
    sample = X.tail(50)
    assert np.allclose(lr.predict_proba(sample), lr2.predict_proba(sample))
    assert np.allclose(boost.predict_proba(sample), boost2.predict_proba(sample))

    print(f"Exported {out_dir}")
    print(f"  logistic regression C={best_c}, {boost_type} with {n_trees} trees, "
          f"{len(features)} features, trained on {len(matches)} matches")
    print("  Loading test passed (reloaded models give identical predictions).")
    if metadata["metrics_final_test_2025_26"] is None:
        print("  Note: final_test.csv not found, so the final-test metrics are missing from metadata.")


if __name__ == "__main__":
    main()
