import asyncio
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config.settings import settings
from app.controllers.match_controller import get_football_api
from app.main import app
from app.services.football_api import FootballAPI, FootballAPIError


SAMPLE_RESPONSE = {
    "matches": [
        {
            "id": 537785,
            "utcDate": "2026-10-03T14:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"id": 57, "name": "Arsenal FC"},
            "awayTeam": {"id": 61, "name": "Chelsea FC"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        },
        {
            "id": 537786,
            "utcDate": "2026-10-04T16:30:00Z",
            "status": "SCHEDULED",
            "homeTeam": {"id": 64, "name": "Liverpool FC"},
            "awayTeam": {"id": 65, "name": "Manchester City FC"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
    ]
}


async def no_sleep(seconds):
    # used instead of asyncio.sleep so retry tests run instantly
    no_sleep.calls.append(seconds)


def make_api(handler):
    no_sleep.calls = []
    return FootballAPI(
        api_key="test-key",
        base_url="https://api.test/v4",
        transport=httpx.MockTransport(handler),
        sleep=no_sleep,
    )


@pytest.fixture
def client():
    yield TestClient(app)
    app.dependency_overrides.clear()


def use_fake_api(handler):
    api = make_api(handler)
    app.dependency_overrides[get_football_api] = lambda: api
    return api


# --- service tests ---------------------------------------------------------

def test_get_matches_normalizes_response():
    seen = {}

    def handler(request):
        seen["url"] = request.url
        seen["token"] = request.headers.get("X-Auth-Token")
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    api = make_api(handler)
    matches = asyncio.run(
        api.get_matches("PL", date(2026, 10, 1), date(2026, 10, 7), "finished")
    )

    assert seen["url"].path == "/v4/competitions/PL/matches"
    assert seen["url"].params["dateFrom"] == "2026-10-01"
    assert seen["url"].params["dateTo"] == "2026-10-07"
    assert seen["url"].params["status"] == "FINISHED"
    assert seen["token"] == "test-key"

    assert matches[0] == {
        "id": 537785,
        "home_team": "Arsenal FC",
        "away_team": "Chelsea FC",
        "date": "2026-10-03T14:00:00Z",
        "status": "FINISHED",
        "home_score": 2,
        "away_score": 1,
    }
    assert matches[1]["home_score"] is None


def test_retries_after_429_then_succeeds():
    responses = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(200, json=SAMPLE_RESPONSE),
    ]
    api = make_api(lambda request: responses.pop(0))

    matches = asyncio.run(api.get_matches())

    assert len(matches) == 2
    assert no_sleep.calls == [3.0]


def test_retry_after_is_capped():
    responses = [
        httpx.Response(429, headers={"Retry-After": "120"}),
        httpx.Response(200, json=SAMPLE_RESPONSE),
    ]
    api = make_api(lambda request: responses.pop(0))

    asyncio.run(api.get_matches())

    assert no_sleep.calls == [10.0]


def test_gives_up_after_three_server_errors():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    api = make_api(handler)

    with pytest.raises(FootballAPIError) as error:
        asyncio.run(api.get_matches())

    assert error.value.status_code == 500
    assert len(calls) == 3


def test_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403, json={"message": "Restricted resource"})

    api = make_api(handler)

    with pytest.raises(FootballAPIError) as error:
        asyncio.run(api.get_matches())

    assert error.value.status_code == 403
    assert len(calls) == 1


def test_missing_api_key_raises_without_calling_api():
    api = FootballAPI(api_key="", transport=httpx.MockTransport(lambda r: 1 / 0))

    with pytest.raises(FootballAPIError) as error:
        asyncio.run(api.get_matches())

    assert "FOOTBALL_API_KEY" in error.value.message


# --- endpoint tests --------------------------------------------------------

def test_endpoint_returns_matches(client):
    use_fake_api(lambda request: httpx.Response(200, json=SAMPLE_RESPONSE))

    response = client.get("/matches/", params={"date_from": "2026-10-01", "date_to": "2026-10-07"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["matches"]) == 2
    assert body["matches"][0]["home_team"] == "Arsenal FC"
    assert body["matches"][0]["home_score"] == 2


def test_endpoint_rejects_bad_date(client):
    use_fake_api(lambda request: httpx.Response(200, json=SAMPLE_RESPONSE))

    response = client.get("/matches/", params={"date_from": "not-a-date"})

    assert response.status_code == 422


def test_endpoint_rejects_date_from_after_date_to(client):
    use_fake_api(lambda request: httpx.Response(200, json=SAMPLE_RESPONSE))

    response = client.get("/matches/", params={"date_from": "2026-10-07", "date_to": "2026-10-01"})

    assert response.status_code == 400


def test_endpoint_without_api_key_returns_503(client, monkeypatch):
    monkeypatch.setattr(settings, "football_api_key", "")

    response = client.get("/matches/")

    assert response.status_code == 503
    assert "FOOTBALL_API_KEY" in response.json()["detail"]


def test_endpoint_retries_429_then_returns_200(client):
    responses = [
        httpx.Response(429),
        httpx.Response(200, json=SAMPLE_RESPONSE),
    ]
    use_fake_api(lambda request: responses.pop(0))

    response = client.get("/matches/")

    assert response.status_code == 200
    assert len(response.json()["matches"]) == 2


def test_endpoint_persistent_server_error_returns_502(client):
    use_fake_api(lambda request: httpx.Response(503))

    response = client.get("/matches/")

    assert response.status_code == 502


def test_endpoint_timeout_returns_503(client):
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    use_fake_api(handler)

    response = client.get("/matches/")

    assert response.status_code == 503
    assert "timed out" in response.json()["detail"]


def test_endpoint_unknown_competition_returns_404(client):
    use_fake_api(lambda request: httpx.Response(404, json={"message": "Not found"}))

    response = client.get("/matches/", params={"competition": "XYZ"})

    assert response.status_code == 404
