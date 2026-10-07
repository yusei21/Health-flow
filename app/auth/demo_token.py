import hmac
from uuid import UUID

from pydantic import SecretStr

from app.auth.principal import Principal
from app.core.exceptions import AuthenticationError


class DemoTokenAuthenticator:
    """DEVELOPMENT-ONLY login: one static bearer token mapped to one demo user.

    Stands in for a real identity provider until Phase F. Disabled when no token is
    configured and refused by Settings in production.
    """

    def __init__(self, token: SecretStr | None, user_id: UUID | None) -> None:
        self._token = token.get_secret_value() if token else ""
        self._user_id = user_id

    def authenticate(self, bearer_token: str | None) -> Principal:
        if not self._token or self._user_id is None or bearer_token is None:
            raise AuthenticationError("demo auth disabled or token missing")
        if not hmac.compare_digest(bearer_token.encode(), self._token.encode()):
            raise AuthenticationError("invalid token")
        return Principal(user_id=self._user_id)
