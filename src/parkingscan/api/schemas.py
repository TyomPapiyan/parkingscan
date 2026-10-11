from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# точка в пикселях исходного кадра и четырёхугольник места
Point = Annotated[list[float], Field(min_length=2, max_length=2)]
Quad = Annotated[list[Point], Field(min_length=4, max_length=4)]


# устройство инференса: идентификатор для torch и человекочитаемое имя
class DeviceInfo(BaseModel):
    id: str
    name: str


# ответ GET /health
class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    models_loaded: bool
    device: DeviceInfo


# источник места по разделу 7.2
SlotSource = Literal["observed", "interpolated", "projected", "extrapolated", "manual"]


# состояние места на снимке; conf - уверенность детекции в занятом месте
class SlotState(BaseModel):
    slot_id: str
    occupied: bool
    confirmed: bool = Field(description="Confirmed by calibration or occupied in this image")
    conf: float | None
    source: SlotSource
    polygon: Quad


# занятость одного ряда
class RowState(BaseModel):
    row: int
    total: int
    occupied: int


# ответ POST /occupancy/detect по разделу 7.3
class OccupancyResponse(BaseModel):
    site_id: str | None = Field(description="Null in one-shot mode: the grid is built from this image")
    timestamp: datetime
    total_slots: int = Field(
        description="All slots of the grid, confirmed_slots + unconfirmed_slots. occupancy_rate is NOT based on it"
    )
    confirmed_slots: int = Field(
        description="Slots confirmed by calibration (both dividers observed or a car seen in the slot) "
        "or occupied in this image"
    )
    unconfirmed_slots: int = Field(
        description="Slots projected by the grid with neither both dividers nor a car; counted neither as free "
        "nor as occupied"
    )
    occupied: int = Field(description="Slots with a car; an occupied slot is always confirmed")
    free: int = Field(description="Confirmed slots without a car: confirmed_slots - occupied")
    occupancy_rate: float | None = Field(
        description="occupied / confirmed_slots; null when there are no confirmed slots or calibration failed"
    )
    cars_in_zone: int = Field(
        description="Detections inside the working zone or inside a slot. When calibration_status is failed "
        "with calibration_reason no_scale there is no working zone and this equals the number of all "
        "detections in the image"
    )
    unassigned_detections: int = Field(description="Detections inside the working zone that match no slot")
    outside_zone_detections: int = Field(description="Detections outside the working zone, not part of occupancy")
    calibration_status: Literal["ok", "partial", "failed", "manual"]
    calibration_reason: str | None = Field(description="Why calibration is partial or failed")
    calibration_coverage: float
    rows: list[RowState]
    slots: list[SlotState]
    inference_ms: int = Field(description="Detection, calibration and slot matching together")


# место в сетке площадки; confirmed - обе границы наблюдаемые либо при калибровке в месте стояла машина
class SlotModel(BaseModel):
    slot_id: str
    row: int
    index: int
    polygon: Quad
    confidence: float
    source: SlotSource
    confirmed: bool


# ряд: разделители по порядку [x_front, y_front, x_back, y_back] и их источники
class RowModel(BaseModel):
    row: int
    dividers: list[list[float]]
    sources: list[str]


# калибровка площадки; в этом же виде хранится в data/sites/{site_id}.json
class SiteCalibration(BaseModel):
    site_id: str
    ref_width: int
    ref_height: int
    images: int
    calibrated_at: datetime
    status: Literal["ok", "partial", "failed", "manual"]
    reason: str | None
    calibration_coverage: float
    zone_polygon: list[Point]
    params: dict[str, float]
    rows: list[RowModel]
    slots: list[SlotModel]


# место при ручной правке сетки
class SlotInput(BaseModel):
    row: int = Field(ge=0)
    index: int = Field(ge=0)
    polygon: Quad


# тело PUT /sites/{site_id}/slots: новая сетка целиком
class SlotsUpdate(BaseModel):
    slots: list[SlotInput] = Field(min_length=1)


# единый формат ошибки для всех эндпоинтов
class ErrorResponse(BaseModel):
    error: str
    detail: str | list[dict[str, Any]]
