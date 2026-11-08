from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


_password_hasher = PasswordHasher()


class AuthenticationService:

    def hash_password(self, password: str) -> str:
        return _password_hasher.hash(password)

    def verify_password(
        self,
        password: str,
        password_hash: str
    ) -> bool:

        try:
            return _password_hasher.verify(password_hash, password)
        except (InvalidHashError, VerificationError):
            return False