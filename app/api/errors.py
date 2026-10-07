import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.core.exceptions import (
    AuthenticationError,
    FacilityProviderError,
    HealthFlowError,
    LLMError,
    MLInferenceError,
    PatientNotFoundError,
)
from app.core.logging import request_id_var

logger = logging.getLogger(__name__)

EMERGENCY_HINT = "Em caso de emergência, ligue 192 (SAMU)."

_STATUS_BY_ERROR: list[tuple[type[HealthFlowError], int]] = [
    (AuthenticationError, status.HTTP_401_UNAUTHORIZED),
    (PatientNotFoundError, status.HTTP_404_NOT_FOUND),
    (LLMError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (MLInferenceError, status.HTTP_503_SERVICE_UNAVAILABLE),
    (FacilityProviderError, status.HTTP_503_SERVICE_UNAVAILABLE),
]


def _body(message: str) -> dict[str, object]:
    return {"error": message, "emergency_hint": EMERGENCY_HINT, "request_id": request_id_var.get()}


async def _handle_domain_error(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HealthFlowError)  # noqa: S101 - registered for this type only
    code = next(
        (code for kind, code in _STATUS_BY_ERROR if isinstance(exc, kind)),
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    logger.warning("request_failed", extra={"error_type": type(exc).__name__, "status": code})
    headers = {"WWW-Authenticate": "Bearer"} if code == status.HTTP_401_UNAUTHORIZED else None
    return JSONResponse(_body(exc.public_message), status_code=code, headers=headers)


async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
    # Never leak stack traces or internals; the traceback goes to server logs only.
    logger.error("unhandled_error", extra={"error_type": type(exc).__name__}, exc_info=exc)
    return JSONResponse(_body("Erro interno."), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(HealthFlowError, _handle_domain_error)
    app.add_exception_handler(Exception, _handle_unexpected)
