from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    Text,
    func,
)

from app.database.database import Base


class User(Base):

    __tablename__ = "users"

    user_id = Column(
        Integer,
        Identity(start=100, minvalue=100, maxvalue=999, always=True),
        primary_key=True,
    )

    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    phone = Column(String(20), unique=True, nullable=True)


class UserPassword(Base):

    __tablename__ = "user_passwords"

    user_id = Column(
        Integer,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    password_hash = Column(Text, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )