"""ML-08: train the main models and compare them with the baselines.

Usage (from ml/pipeline):
    pip install xgboost
    python train_models.py

Reads  data/clean/features.csv   (run features.py first)
Writes data/clean/model_predictions_val.csv  (validation-season probabilities per model)
       data/results/baseline_metrics.csv     (adds one row per model)

Models: multinomial logistic regression, random forest, XGBoost, and an average of
logistic regression + XGBoost. All are trained on seasons up to TRAIN_END and scored on
the validation season. The test season is NOT used.

Choices made WITHOUT looking at the validation season:
 - Logistic regression's strength setting (C) and XGBoost's number of trees are picked
   using the last training season as a mini validation set, then the model is re-fitted
   on all training seasons.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (CLEAN_DIR, RESULTS_DIR, TRAIN_END, VAL_SEASON, log_loss, report,
                    split_masks)

try:
    from xgboost import XGBClassifier
except ImportError:  # fall back so the script still runs
    XGBClassifier = None

ID_COLS = ["Div", "Date", "Season", "HomeTeam", "AwayTeam"]


def make_lr(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000))


def xgb_params():
    return dict(objective="multi:softprob", learning_rate=0.03, max_depth=3, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0, random_state=0, n_jobs=-1)


def main():
    path = CLEAN_DIR / "features.csv"
    if not path.exists():
        raise SystemExit("features.csv not found. Run features.py first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    features = [c for c in df.columns if c not in ID_COLS + ["y"]]
    train_m, val_m, _ = split_masks(df)

    # Mini validation set inside the training data: the last training season
    last_train_season = sorted(df.loc[train_m, "Season"].unique())[-1]
    inner_val_m = train_m & (df["Season"] == last_train_season)
    inner_train_m = train_m & ~inner_val_m

    X, y = df[features], df["y"].astype(int)
    X_tr, y_tr = X[train_m], y[train_m]
    X_in_tr, y_in_tr = X[inner_train_m], y[inner_train_m]
    X_in_va, y_in_va = X[inner_val_m], y[inner_val_m]
    X_va, y_va = X[val_m], y[val_m].to_numpy()

    print(f"{len(features)} features | train {train_m.sum()} matches | "
          f"validation {val_m.sum()} matches ({VAL_SEASON})")
    print(f"Mini validation season used for choosing settings: {last_train_season}\n")

    preds = {}

    # --- Logistic regression: pick C on the mini validation set ---
    best_c, best_ll = None, np.inf
    for C in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        m = make_lr(C).fit(X_in_tr, y_in_tr)
        ll = log_loss(m.predict_proba(X_in_va), y_in_va.to_numpy())
        print(f"  logistic regression C={C:<6} mini-validation log loss {ll:.4f}")
        if ll < best_ll:
            best_c, best_ll = C, ll
    lr = make_lr(best_c).fit(X_tr, y_tr)
    preds["Logistic regression"] = lr.predict_proba(X_va)
    print(f"  -> chose C={best_c}\n")

    # --- Random forest ---
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=400, min_samples_leaf=25,
                                              max_features="sqrt", n_jobs=-1, random_state=0))
    rf.fit(X_tr, y_tr)
    preds["Random forest"] = rf.predict_proba(X_va)

    # --- Gradient boosting: XGBoost (or sklearn fallback) ---
    importances = None
    if XGBClassifier is not None:
        probe = XGBClassifier(n_estimators=1000, early_stopping_rounds=50,
                              eval_metric="mlogloss", **xgb_params())
        probe.fit(X_in_tr, y_in_tr, eval_set=[(X_in_va, y_in_va)], verbose=False)
        n_trees = int(probe.best_iteration) + 1
        print(f"  XGBoost: {n_trees} trees chosen on the mini validation set\n")
        xgb = XGBClassifier(n_estimators=n_trees, eval_metric="mlogloss", **xgb_params())
        xgb.fit(X_tr, y_tr)
        gb_name = "XGBoost"
        importances = pd.Series(xgb.feature_importances_, index=features)
    else:
        print("  xgboost is not installed (pip install xgboost); using scikit-learn's "
              "gradient boosting instead.\n")
        xgb = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_depth=3,
                                             l2_regularization=5.0, random_state=0)
        xgb.fit(X_tr, y_tr)
        gb_name = "Gradient boosting (sklearn)"
    preds[gb_name] = xgb.predict_proba(X_va)

    preds[f"LR + {gb_name} (average)"] = (preds["Logistic regression"] + preds[gb_name]) / 2

    # --- Report ---
    print("RESULTS ON THE VALIDATION SEASON")
    print("-" * 100)
    for name, p in preds.items():
        report(name, VAL_SEASON, p, y_va)

    out = df.loc[val_m, ID_COLS].copy()
    for name, p in preds.items():
        tag = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
        out[f"{tag}_p_home"], out[f"{tag}_p_draw"], out[f"{tag}_p_away"] = p.round(4).T
    out.to_csv(CLEAN_DIR / "model_predictions_val.csv", index=False)

    # --- Full comparison table ---
    table = pd.read_csv(RESULTS_DIR / "baseline_metrics.csv")
    table = table[table["season"] == VAL_SEASON].sort_values("log_loss", na_position="last")
    print(f"\nALL MODELS ON {VAL_SEASON} (best log loss first)")
    print(table[["model", "log_loss", "brier", "rps", "accuracy"]].to_string(index=False))

    if importances is not None:
        print("\nMost important features for XGBoost:")
        print(importances.sort_values(ascending=False).head(15).round(3).to_string())

    print(f"\nSaved {CLEAN_DIR / 'model_predictions_val.csv'}")
    print("The test season has not been touched.")


if __name__ == "__main__":
    main()
