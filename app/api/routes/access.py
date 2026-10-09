"""Capabilities are explicit: this prototype cannot authenticate with gov.br yet."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/access", tags=["access"])


@router.get("/capabilities")
async def capabilities(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    return {
        "govbr_available": False,
        "record_source": "synthetic_demo",
        "demo_available": settings.demo_auth_enabled and settings.demo_user_id is not None,
        "audio_available": bool(settings.transcription_url),
    }
