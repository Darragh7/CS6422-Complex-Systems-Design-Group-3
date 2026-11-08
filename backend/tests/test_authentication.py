import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.database import Base, get_db
from app.main import app
from app.models.user import User, UserPassword
from app.services.authentication_service import AuthenticationService


@pytest.fixture
def auth_client():
    client, engine = make_client()
    yield client, engine
    app.dependency_overrides.clear()
    engine.dispose()


def test_passwords_are_hashed_and_verified():
    service = AuthenticationService()
    password_hash = service.hash_password("correct horse battery staple")

    assert password_hash.startswith("$argon2id$")
    assert password_hash != "correct horse battery staple"
    assert service.verify_password("correct horse battery staple", password_hash)
    assert not service.verify_password("wrong password", password_hash)


def test_invalid_password_hash_is_rejected():
    assert not AuthenticationService().verify_password("password", "not-a-hash")


def test_signup_persists_user_and_password_hash(auth_client):
    client, engine = auth_client
    response = client.post(
        "/users/signup",
        json={
            "first_name": "Ada",
            "last_name": "Lovelace",
            "email": "ADA@example.com",
            "password": "correct horse battery staple",
        },
    )

    assert response.status_code == 201
    user_data = response.json()["user"]
    assert user_data["user_id"] >= 1
    assert user_data["email"] == "ada@example.com"
    assert "password" not in user_data

    with sessionmaker(bind=engine)() as db:
        user = db.query(User).filter_by(user_id=user_data["user_id"]).one()
        password = db.query(UserPassword).filter_by(user_id=user.user_id).one()
        assert user.first_name == "Ada"
        assert password.password_hash.startswith("$argon2id$")
        assert password.password_hash != "correct horse battery staple"

def test_login_accepts_email_case_insensitively_and_rejects_bad_password(auth_client):
    client, engine = auth_client
    signup = client.post(
        "/users/signup",
        json={
            "first_name": "Ada",
            "last_name": "Lovelace",
            "email": "ada@example.com",
            "password": "correct horse battery staple",
        },
    )
    assert signup.status_code == 201

    response = client.post(
        "/users/login",
        json={
            "identifier": "ADA@EXAMPLE.COM",
            "password": "correct horse battery staple",
        },
    )
    assert response.status_code == 200
    assert response.json()["user"]["email"] == "ada@example.com"

    response = client.post(
        "/users/login",
        json={"identifier": "ada@example.com", "password": "incorrect password"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email, phone, or password"

def test_signup_rejects_duplicate_contact_and_missing_contact(auth_client):
    client, _ = auth_client
    user_data = {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "email": "ada@example.com",
        "password": "correct horse battery staple",
    }

    assert client.post("/users/signup", json=user_data).status_code == 201
    assert client.post("/users/signup", json=user_data).status_code == 409

    no_contact = {**user_data, "email": ""}
    response = client.post("/users/signup", json=no_contact)
    assert response.status_code == 422


def make_client():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    test_session = sessionmaker(bind=engine)

    def override_get_db():
        db = test_session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    return client, engine