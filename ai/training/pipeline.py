"""ScoreLine end-to-end skeleton pipeline. Stage 1 (objective/metrics) is CONFIG."""
from . import preprocessing as pp
from . import train as tr

CONFIG = {
    "target": "result",  # H / D / A
    "metrics": ["log_loss", "accuracy"],
    "baseline": "always_home",
    "test_frac": 0.2,
    "cv_splits": 3,
}


def load_historical(competition: str = "PL", seasons=(2024,), **kwargs) -> list[dict]:
    """Stage 2 option: real finished matches from football-data.org (cached CSV)."""
    from .data_pull import pull_matches

    return pull_matches(competition, list(seasons), **kwargs).to_dict("records")


def run(matches: list[dict], config: dict = CONFIG) -> dict:
    """Chain stages: parse -> inspect -> clean -> label -> features (prior-only) -> split -> select model -> evaluate."""
    df = pp.load_api_matches(matches)
    pp.inspect(df)
    df = pp.label_result(pp.clean(df))
    # Features use prior matches only, so build on the full timeline, then split.
    df = df.join(tr.build_features(df))
    train_df, test_df = pp.time_split(df, config["test_frac"])
    # Stage 6 (explore) happens in notebooks on train_df only.
    X_train, y_train = train_df[tr.FEATURE_COLUMNS], train_df[config["target"]]
    best, model, scores = tr.select_model(X_train, y_train, n_splits=config["cv_splits"])
    result = tr.evaluate(model, test_df[tr.FEATURE_COLUMNS], test_df[config["target"]], y_train)
    return {**result, "selected_model": best, "selection_scores": scores}
