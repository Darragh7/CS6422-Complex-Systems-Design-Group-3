from fastapi import APIRouter

from app.services.prediction_service import PredictionService


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)


@router.get("/{match_id}")
def get_prediction(match_id: int):

    prediction_service = PredictionService()

    prediction = prediction_service.predict(match_id)

    return {
        "match_id": match_id,
        "prediction": prediction
    }