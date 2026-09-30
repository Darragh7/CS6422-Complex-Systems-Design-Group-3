from fastapi import FastAPI

from app.controllers.match_controller import router as match_router
from app.controllers.prediction_controller import router as prediction_router

app = FastAPI(
    title="ScoreLine",
    description="Backend API",
    version="0.1.0",
)

app.include_router(match_router)
app.include_router(prediction_router)


@app.get("/")
def root():
    return {
        "message": "Scoreline Backend Active",
        "version": "0.1.0"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }