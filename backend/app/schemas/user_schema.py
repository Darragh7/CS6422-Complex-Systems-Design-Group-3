import re
from typing import Any

from django.core.exceptions import ValidationError as DjangoValidationError
from django.core.validators import validate_email
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UserCreate(BaseModel):

    first_name: str = Field(min_length=1, max_length=50)
    last_name: str = Field(min_length=1, max_length=50)
    email: str | None = Field(default=None, max_length=100)
    phone: str | None = Field(default=None, max_length=20)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("first_name", "last_name", mode="before")
    @classmethod
    def trim_name(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip().lower()
            return value or None
        return value

    @field_validator("email")
    @classmethod
    def validate_email_address(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                validate_email(value)
            except DjangoValidationError as exc:
                raise ValueError("Enter a valid email address") from exc
        return value

    @field_validator("phone", mode="before")
    @classmethod
    def normalize_phone(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None

            # Accept common human formatting, then store the exact canonical
            # format enforced by database/schema.sql.
            compact = re.sub(r"[\s().-]", "", value)
            if compact.startswith("+353"):
                return "+353 " + compact[4:]
            if compact.startswith("+91"):
                return "+91 " + compact[3:]
            return compact
        return value

    @field_validator("phone")
    @classmethod
    def validate_phone_number(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(
            r"(?:\+353 8[2-9][0-9]{7}|\+91 [6-9][0-9]{9})",
            value,
        ):
            raise ValueError("Enter a supported Irish or Indian phone number")
        return value

    @model_validator(mode="after")
    def require_contact_method(self):
        if self.email is None and self.phone is None:
            raise ValueError("Provide an email address or phone number")
        if not self.first_name or not self.last_name:
            raise ValueError("First and last names are required")
        return self


class UserLogin(BaseModel):

    identifier: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("identifier")
    @classmethod
    def normalize_identifier(cls, value: str) -> str:
        value = value.strip()
        if "@" in value:
            return value.lower()
        return value


class UserResponse(BaseModel):

    model_config = ConfigDict(from_attributes=True)

    user_id: int
    first_name: str
    last_name: str
    email: str | None
    phone: str | None


class AuthResponse(BaseModel):

    user: UserResponse
