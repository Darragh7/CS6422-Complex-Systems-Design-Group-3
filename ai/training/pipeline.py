"""ScoreLine end-to-end skeleton pipeline. Stage 1 (objective/metrics) is CONFIG."""
from . import preprocessing as pp
from . import train as tr

CONFIG = {
    "target": "result",  # H / D / A
    "metrics": ["log_loss", "accuracy"],
    "baseline": "always_home",
    "test_frac": 0.2,
}


def run(matches: list[dict], config: dict = CONFIG) -> dict:
    """Chain stages: parse -> inspect -> clean -> label -> split -> features -> train -> evaluate."""
    df = pp.load_api_matches(matches)
    pp.inspect(df)
    df = pp.label_result(pp.clean(df))
    train_df, test_df = pp.time_split(df, config["test_frac"])
    # Stage 6 (explore) happens in notebooks on train_df only.
    model = tr.train_model(tr.build_features(train_df), train_df[config["target"]])
    return tr.evaluate(model, tr.build_features(test_df), test_df[config["target"]])
