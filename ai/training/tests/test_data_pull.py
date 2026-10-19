import httpx
import pytest

from ai.training import data_pull


def api_match(i, home=2, away=1):
    return {"id": i, "utcDate": f"2024-08-{i:02d}T14:00:00Z", "status": "FINISHED",
            "homeTeam": {"name": f"Home {i}"}, "awayTeam": {"name": f"Away {i}"},
            "score": {"fullTime": {"home": home, "away": away}}}


def mock_client(handler):
    return httpx.Client(base_url="https://example.test/v4", headers={"X-Auth-Token": "k"},
                        transport=httpx.MockTransport(handler))


def test_fetch_season_normalises_and_sends_params():
    seen = {}

    def handler(request):
        seen["path"], seen["params"] = request.url.path, dict(request.url.params)
        seen["token"] = request.headers["X-Auth-Token"]
        return httpx.Response(200, json={"matches": [api_match(1), api_match(2, 0, 0)]})

    rows = data_pull.fetch_season(mock_client(handler), "pl", 2024)
    assert seen == {"path": "/v4/competitions/PL/matches",
                    "params": {"season": "2024", "status": "FINISHED"}, "token": "k"}
    assert rows[0] == {"id": 1, "home_team": "Home 1", "away_team": "Away 1",
                       "date": "2024-08-01T14:00:00Z", "status": "FINISHED",
                       "home_score": 2, "away_score": 1}


def test_fetch_season_error_raises():
    client = mock_client(lambda r: httpx.Response(403, json={"message": "restricted"}))
    with pytest.raises(data_pull.DataPullError, match="403 restricted"):
        data_pull.fetch_season(client, "PL", 2015)


def test_fetch_season_retries_once_on_429():
    responses = iter([httpx.Response(429, headers={"Retry-After": "1"}),
                      httpx.Response(200, json={"matches": [api_match(3)]})])
    waits = []
    rows = data_pull.fetch_season(mock_client(lambda r: next(responses)), "PL", 2024, sleep=waits.append)
    assert len(rows) == 1 and waits == [1.0]


def test_pull_matches_caches_and_rate_limits(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.params["season"])
        return httpx.Response(200, json={"matches": [api_match(len(calls))]})

    waits = []
    client = mock_client(handler)
    df = data_pull.pull_matches("PL", [2023, 2024], data_dir=tmp_path, client=client, sleep=waits.append)
    assert len(df) == 2 and calls == ["2023", "2024"] and waits == [data_pull.MIN_INTERVAL_SECONDS]
    assert (tmp_path / "PL_2023.csv").exists()

    df2 = data_pull.pull_matches("PL", [2023, 2024], data_dir=tmp_path, client=client, sleep=waits.append)
    assert len(df2) == 2 and len(calls) == 2  # served from cache


def test_missing_key(monkeypatch):
    monkeypatch.delenv("FOOTBALL_API_KEY", raising=False)
    with pytest.raises(data_pull.DataPullError):
        data_pull.make_client()
