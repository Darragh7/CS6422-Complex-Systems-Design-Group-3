"""FINAL TEST: score the finished ensemble once on the untouched test season (2025/26).

Usage (from ml/pipeline):
    python final_test.py --confirm

RUN THIS ONCE, after you have decided on the final model. Do not change anything and re-run
it to get a better number: that would turn the test season into a second validation season
and the result would no longer be honest. Report whatever number comes out.

What it does (same recipe as before, shifted one season forward):
 - Logistic regression and XGBoost are trained on seasons up to 2024/25, with their settings
   chosen on 2024/25 as the mini validation set.
 - Elo uses the pre-match ratings already saved in elo_features.csv (past matches only).
 - Poisson is re-fitted weekly through 2025/26 using only earlier matches (walk-forward).
 - The ensemble is the plain average of those four, compared with the bookmakers.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import poisson_baseline as pb
import train_models as tm
from common import CLEAN_DIR, RESULTS_DIR, TEST_SEASON, VAL_SEASON, load_matches, log_loss, report
from evaluate import ENSEMBLE, KEYS, MARKET, bootstrap_ci, per_match_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true",
                        help="I understand this is the one-time final test")
    args = parser.parse_args()
    if not args.confirm:
        print(__doc__)
        print("Nothing was run. Add --confirm when you are ready for the one-time final test.")
        return

    matches = load_matches()
    feats = pd.read_csv(CLEAN_DIR / "features.csv", parse_dates=["Date"])
    elo = pd.read_csv(CLEAN_DIR / "elo_features.csv", parse_dates=["Date"])
    for name, other in (("features.csv", feats), ("elo_features.csv", elo)):
        if not matches[KEYS].reset_index(drop=True).equals(other[KEYS].reset_index(drop=True)):
            raise SystemExit(f"{name} does not line up with matches.csv. Re-run clean.py, "
                             f"elo_baseline.py and features.py in order.")

    test_m = (matches["Season"] == TEST_SEASON).to_numpy()
    fit_m = (matches["Season"] <= VAL_SEASON).to_numpy()  # train + validation seasons
    inner_val = fit_m & (matches["Season"] == VAL_SEASON).to_numpy()
    inner_tr = fit_m & ~inner_val
    y_all = matches["y"].to_numpy().astype(int)
    y_test = y_all[test_m]
    print(f"Fit on seasons up to {VAL_SEASON} ({fit_m.sum()} matches); "
          f"test on {TEST_SEASON} ({test_m.sum()} matches)\n")

    # ---- Logistic regression and boosting ----
    features = [c for c in feats.columns if c not in tm.ID_COLS + ["y"]]
    X, y = feats[features], feats["y"].astype(int)

    best_c, best_ll = None, np.inf
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        m = tm.make_lr(C).fit(X[inner_tr], y[inner_tr])
        ll = log_loss(m.predict_proba(X[inner_val]), y[inner_val].to_numpy())
        if ll < best_ll:
            best_c, best_ll = C, ll
    lr = tm.make_lr(best_c).fit(X[fit_m], y[fit_m])
    p_lr = lr.predict_proba(X[test_m])
    print(f"Logistic regression: C={best_c}")

    if tm.XGBClassifier is not None:
        probe = tm.XGBClassifier(n_estimators=1000, early_stopping_rounds=50,
                                 eval_metric="mlogloss", **tm.xgb_params())
        probe.fit(X[inner_tr], y[inner_tr], eval_set=[(X[inner_val], y[inner_val])], verbose=False)
        n_trees = int(probe.best_iteration) + 1
        boost = tm.XGBClassifier(n_estimators=n_trees, eval_metric="mlogloss", **tm.xgb_params())
        boost.fit(X[fit_m], y[fit_m])
        print(f"XGBoost: {n_trees} trees")
    else:
        print("xgboost not installed: using scikit-learn gradient boosting instead")
        boost = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_depth=3,
                                               l2_regularization=5.0, random_state=0).fit(X[fit_m], y[fit_m])
    p_boost = boost.predict_proba(X[test_m])

    # ---- Elo (pre-match values already saved) ----
    p_elo = elo.loc[test_m, ["elo_p_home", "elo_p_draw", "elo_p_away"]].to_numpy()

    # ---- Poisson (weekly walk-forward over the test season) ----
    X_p, y_p, _ = pb.build_design(matches)
    day_match = (matches["Date"] - matches["Date"].min()).dt.days.to_numpy()
    lam, _, _ = pb.walk_forward(X_p, y_p, day_match, np.flatnonzero(test_m), pb.ALPHA, pb.HALF_LIFE_DAYS)
    p_poi = pb.outcome_probs(lam[:, 0], lam[:, 1])

    models = {
        "Elo": p_elo, "Poisson": p_poi, "Logistic regression": p_lr,
        "XGBoost" if tm.XGBClassifier is not None else "Gradient boosting (sklearn)": p_boost,
        ENSEMBLE: (p_elo + p_poi + p_lr + p_boost) / 4,
    }

    # ---- Bookmakers ----
    sub = np.ones(test_m.sum(), dtype=bool)
    odds_path = CLEAN_DIR / "odds.csv"
    if odds_path.exists():
        odds = pd.read_csv(odds_path, parse_dates=["Date"])
        merged = matches.loc[test_m, KEYS].merge(
            odds[KEYS + ["market_p_home", "market_p_draw", "market_p_away"]], on=KEYS, how="left")
        sub = merged["market_p_home"].notna().to_numpy()
        models[MARKET] = np.nan_to_num(
            merged[["market_p_home", "market_p_draw", "market_p_away"]].to_numpy(), nan=1 / 3)
    else:
        print("(No odds.csv: run odds.py for the bookmaker comparison.)")

    label = f"{TEST_SEASON} (final test)"
    y_s = y_test[sub]
    print(f"\nFINAL TEST RESULTS on {label}: {sub.sum()} matches")
    print("-" * 100)
    rows = []
    for name, p in models.items():
        rows.append(report(name, label, p[sub], y_s))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("log_loss").to_csv(RESULTS_DIR / "final_test.csv", index=False)

    # ---- Is the ensemble really different? ----
    rng = np.random.default_rng(0)
    loss = {n: per_match_loss(p[sub], y_s) for n, p in models.items()}
    print("\nENSEMBLE vs REFERENCE (average log-loss difference, 95% interval; negative = ensemble better)")
    for ref in ["Elo"] + ([MARKET] if MARKET in models else []):
        mean, lo, hi = bootstrap_ci(loss[ENSEMBLE] - loss[ref], rng)
        verdict = "better" if hi < 0 else ("worse" if lo > 0 else "tied")
        print(f"   vs {ref:<30} {mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    print(f"\nSaved {RESULTS_DIR / 'final_test.csv'}")
    print("This is your honest final number. Do not re-run to improve it.")


if __name__ == "__main__":
    main()
