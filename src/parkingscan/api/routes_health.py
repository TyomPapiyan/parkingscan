from fastapi import APIRouter, Request

from parkingscan import __version__
from parkingscan.api.deps import DeviceDep
from parkingscan.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


# статус сервиса, версия, загружены ли модели и устройство инференса
@router.get("/health")
async def health(request: Request, device: DeviceDep) -> HealthResponse:
    models_loaded = getattr(request.app.state, "detector", None) is not None
    return HealthResponse(status="ok", version=__version__, models_loaded=models_loaded, device=device)
