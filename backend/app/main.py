import logging

from fastapi import FastAPI, HTTPException
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from app.controllers.match_controller import router as match_router
from app.controllers.prediction_controller import router as prediction_router
from app.controllers.user_controller import router as user_router
from app.database.database import engine


logger = logging.getLogger(__name__)

app = FastAPI(
    title="ScoreLine",
    description="Backend API",
    version="0.1.0",
)

app.include_router(match_router)
app.include_router(prediction_router)
app.include_router(user_router)


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
            inspector = inspect(connection)
            required_tables = {"users", "user_passwords"}
            missing_tables = required_tables.difference(inspector.get_table_names())
            if missing_tables:
                logger.error(
                    "Database is missing required authentication tables: %s",
                    ", ".join(sorted(missing_tables)),
                )
                raise HTTPException(
                    status_code=503,
                    detail="Database schema unavailable",
                )
    except HTTPException:
        raise
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
