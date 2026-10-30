import logging

from fastapi import FastAPI, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.controllers.match_controller import router as match_router
from app.controllers.prediction_controller import router as prediction_router
from app.database.database import engine


logger = logging.getLogger(__name__)

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


@app.get("/ready")
def readiness_check():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        logger.exception("Database readiness check failed")
        raise HTTPException(
            status_code=503,
            detail="Database connection unavailable",
        ) from exc

    return {
        "status": "ready",
        "database": "connected",
    }