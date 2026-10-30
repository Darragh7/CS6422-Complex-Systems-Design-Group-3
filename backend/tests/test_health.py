from unittest.mock import MagicMock

from sqlalchemy.exc import OperationalError
from fastapi.testclient import TestClient

from app.database.database import engine
from app.main import app


def test_health_check_does_not_require_database():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_readiness_check_confirms_database_connection(monkeypatch):
    connection = MagicMock()
    monkeypatch.setattr(engine, "connect", lambda: connection)

    response = TestClient(app).get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "database": "connected"}
    connection.__enter__.return_value.execute.assert_called_once()


def test_readiness_check_returns_unavailable_when_database_is_down(monkeypatch):
    def unavailable():
        raise OperationalError("SELECT 1", {}, Exception("offline"))

    monkeypatch.setattr(engine, "connect", unavailable)

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Database connection unavailable"}
