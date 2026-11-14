import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.prediction_schema import PredictionResponse
from app.services.prediction_service import PredictionService


def test_prediction_service_returns_outcome_probabilities():
    prediction = PredictionService().predict(match_id=42)

    assert prediction == {
        "home_win": 0.45,
        "draw": 0.30,
        "away_win": 0.25,
    }


def test_prediction_endpoint_includes_requested_match_id():
    response = TestClient(app).get("/predictions/42")

    assert response.status_code == 200
    assert response.json() == {
        "match_id": 42,
        "prediction": {
            "home_win": 0.45,
            "draw": 0.30,
            "away_win": 0.25,
        },
    }


def test_prediction_endpoint_rejects_non_integer_match_id():
    response = TestClient(app).get("/predictions/not-an-id")

    assert response.status_code == 422


@pytest.mark.parametrize(
    "probabilities",
    [
        {"home_win": -0.1, "draw": 0.5, "away_win": 0.6},
        {"home_win": 0.4, "draw": 1.1, "away_win": 0.2},
        {"home_win": 0.3, "draw": 0.3, "away_win": -0.1},
    ],
)
def test_prediction_response_rejects_probabilities_outside_zero_to_one(probabilities):
    with pytest.raises(ValidationError):
        PredictionResponse(**probabilities)


def test_prediction_response_accepts_boundary_probabilities():
    prediction = PredictionResponse(home_win=0, draw=1, away_win=0)

    assert prediction.model_dump() == {"home_win": 0, "draw": 1, "away_win": 0}
