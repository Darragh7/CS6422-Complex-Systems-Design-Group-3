from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session
import logging

from app.database.database import get_db
from app.models.user import User, UserPassword
from app.schemas.user_schema import AuthResponse, UserCreate, UserLogin
from app.services.authentication_service import AuthenticationService


router = APIRouter(prefix="/users", tags=["Users"])
logger = logging.getLogger(__name__)
authentication_service = AuthenticationService()
_DUMMY_PASSWORD_HASH = authentication_service.hash_password("timing-only")


def _log_database_failure(message, error):
    original = getattr(error, "orig", None)
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    diagnostic = getattr(original, "diag", None)
    constraint = getattr(diagnostic, "constraint_name", None)
    logger.error(
        "%s (error_type=%s, sqlstate=%s, constraint=%s)",
        message,
        type(error).__name__,
        sqlstate or "unknown",
        constraint or "unknown",
    )


@router.post(
    "/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
)
def sign_up(user_data: UserCreate, db: Session = Depends(get_db)):
    contact_filters = []
    if user_data.email:
        contact_filters.append(func.lower(User.email) == user_data.email)
    if user_data.phone:
        contact_filters.append(User.phone == user_data.phone)

    try:
        existing_contact = (
            db.query(User.user_id).filter(or_(*contact_filters)).first()
        )
        if existing_contact:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with that email or phone already exists",
            )

        user = User(
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            email=user_data.email,
            phone=user_data.phone,
        )
        password_hash = authentication_service.hash_password(user_data.password)
        db.add(user)
        db.flush()
        db.add(
            UserPassword(
                user_id=user.user_id,
                password_hash=password_hash,
            )
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        try:
            duplicate_contact = (
                db.query(User.user_id).filter(or_(*contact_filters)).first()
            )
        except SQLAlchemyError as lookup_error:
            _log_database_failure("Signup duplicate check failed", lookup_error)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Account service temporarily unavailable",
            ) from lookup_error
        if duplicate_contact:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with that email or phone already exists",
            ) from exc

        _log_database_failure("Signup database integrity error", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create account",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        _log_database_failure("Signup database operation failed", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Account service temporarily unavailable",
        ) from exc

    db.refresh(user)
    return {"user": user}


@router.post("/login", response_model=AuthResponse)
def log_in(credentials: UserLogin, db: Session = Depends(get_db)):
    identifier = credentials.identifier
    identity_filters = []
    if "@" in identifier:
        identity_filters.append(func.lower(User.email) == identifier)
    else:
        identity_filters.append(User.phone == identifier)
    user_record = (
        db.query(User, UserPassword)
        .join(UserPassword, User.user_id == UserPassword.user_id)
        .filter(or_(*identity_filters))
        .first()
    )

    password_hash = (
        user_record.UserPassword.password_hash if user_record else _DUMMY_PASSWORD_HASH
    )
    password_is_valid = authentication_service.verify_password(
        credentials.password,
        password_hash,
    )
    if not user_record or not password_is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email, phone, or password",
        )

    return {"user": user_record.User}
