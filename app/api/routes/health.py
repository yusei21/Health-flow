from fastapi import APIRouter

from app.api.dependencies.providers import ContainerDep

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(container: ContainerDep) -> dict[str, object]:
    return {"status": "ok", "ml_model_loaded": container.ml_available}
