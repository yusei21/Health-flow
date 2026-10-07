from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.dependencies.container import Container
from app.auth.principal import Principal

_bearer = HTTPBearer(auto_error=False)


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


def get_current_principal(
    container: Annotated[Container, Depends(get_container)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    return container.authenticator.authenticate(credentials.credentials if credentials else None)


ContainerDep = Annotated[Container, Depends(get_container)]
PrincipalDep = Annotated[Principal, Depends(get_current_principal)]
