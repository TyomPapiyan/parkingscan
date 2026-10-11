import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from parkingscan import __version__
from parkingscan.api import routes_health, routes_occupancy, routes_sites
from parkingscan.api.schemas import DeviceInfo, ErrorResponse
from parkingscan.config import get_settings, load_inference_config
from parkingscan.core.detector import CarDetector, select_device
from parkingscan.logging_conf import setup_logging

logger = logging.getLogger(__name__)


# всё тяжёлое загружается один раз при старте и живёт в app.state до остановки
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    config = load_inference_config(settings.inference_config_path)
    device_id, device_name = select_device()
    app.state.device = DeviceInfo(id=device_id, name=device_name)
    start = time.perf_counter()
    app.state.detector = CarDetector(settings.models_dir / config.model_file, config, device_id)
    app.state.detector.warmup()
    logger.info(
        "ParkingScan %s started, device %s (%s), detector loaded and warmed up in %.1f s",
        __version__,
        device_id,
        device_name,
        time.perf_counter() - start,
    )
    yield


app = FastAPI(title="ParkingScan", version=__version__, lifespan=lifespan)
app.include_router(routes_health.router)
app.include_router(routes_occupancy.router)
app.include_router(routes_sites.router)


# HTTP-ошибки отдаём в едином формате error/detail
@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    body = ErrorResponse(error=HTTPStatus(exc.status_code).phrase, detail=str(exc.detail))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump(), headers=exc.headers)


# ошибки валидации запроса тоже в формате error/detail, detail - список полей с причинами
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    body = ErrorResponse(error="Validation Error", detail=jsonable_encoder(exc.errors()))
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=body.model_dump())
