from ai.training.pipeline import run


def test_pipeline_smoke():
    results = ["H", "D", "A", "H", "H"] * 4
    matches = [
        {"id": i, "home_team": "X", "away_team": "Y",
         "date": f"2026-01-{i + 1:02d}T15:00:00Z", "status": "FINISHED",
         "home_score": {"H": 2, "D": 1, "A": 0}[r], "away_score": 1 if r != "A" else 2}
        for i, r in enumerate(results)
    ]
    matches.append({"id": 99, "home_team": "X", "away_team": "Y", "date": "2026-02-01T15:00:00Z",
                    "status": "SCHEDULED", "home_score": None, "away_score": None})
    out = run(matches)
    assert set(out) >= {"log_loss", "accuracy", "baseline_home_accuracy"}
    assert 0 <= out["accuracy"] <= 1
