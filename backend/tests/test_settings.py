import pytest
from pydantic import ValidationError

from app.config.settings import Settings


def test_settings_require_database_url_and_secret_key(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)

    missing_fields = {entry["loc"][0] for entry in error.value.errors()}
    assert missing_fields == {"database_url", "secret_key"}


@pytest.mark.parametrize("field", ["database_url", "secret_key"])
def test_settings_reject_empty_required_values(field):
    values = {
        "database_url": "sqlite://",
        "secret_key": "test-secret",
    }
    values[field] = ""

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)
