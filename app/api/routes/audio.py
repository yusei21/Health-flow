"""Opt-in, bounded, memory-only upload to a configured transcription service."""

import httpx
from fastapi import APIRouter, HTTPException, Request, Response

from app.api.dependencies.providers import PrincipalDep

router = APIRouter(prefix="/audio", tags=["audio"])
MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4"}


@router.post("/transcribe")
async def transcribe(
    request: Request, response: Response, principal: PrincipalDep, consent: bool = False
) -> dict[str, str]:
    if not consent:
        raise HTTPException(403, "Autorize o envio do áudio para transcrição.")
    settings = request.app.state.settings
    if not settings.transcription_url:
        raise HTTPException(503, "Transcrição de áudio não configurada. Use texto.")
    mime = request.headers.get("content-type", "").split(";")[0]
    if mime not in AUDIO_TYPES:
        raise HTTPException(415, "Formato de áudio não suportado.")
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Áudio deve ter no máximo 10 MB.")
        data.extend(chunk)
    if not data:
        raise HTTPException(422, "Áudio vazio.")
    extensions = {"audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/x-wav": "wav"}
    extension = extensions.get(mime, mime.split("/")[1])
    headers = (
        {"Authorization": f"Bearer {settings.transcription_api_key.get_secret_value()}"}
        if settings.transcription_api_key
        else {}
    )
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            result = await client.post(
                settings.transcription_url,
                headers=headers,
                files={"file": (f"report.{extension}", bytes(data), mime)},
                data={"model": settings.transcription_model, "language": "pt"},
            )
        result.raise_for_status()
        payload = result.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(503, "Transcrição indisponível. Use texto.") from exc
    text = payload.get("text") if isinstance(payload, dict) else None
    if not isinstance(text, str) or not 3 <= len(text.strip()) <= 2000:
        raise HTTPException(422, "Transcrição inválida ou longa. Divida o áudio ou use texto.")
    response.headers["Cache-Control"] = "no-store"
    return {"text": text.strip()}
