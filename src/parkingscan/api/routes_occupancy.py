import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status

from parkingscan.api.deps import DetectorDep, SettingsDep, read_image
from parkingscan.api.routes_sites import load_site, slot_models
from parkingscan.api.schemas import OccupancyResponse, RowState, SlotModel, SlotState
from parkingscan.config import Settings
from parkingscan.core.calibration import calibrate
from parkingscan.core.detector import MODEL_CONF, CarDetector
from parkingscan.core.occupancy import Assignment, Summary, assign_detections, summarize
from parkingscan.core.render import ImageFormat, encode_image, fit_max_side, render_occupancy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/occupancy", tags=["occupancy"])

# загружаемое изображение, поле multipart-формы
ImageFile = Annotated[UploadFile, File(description="Aerial image of the parking lot")]

# порог уверенности из запроса; если не задан, берётся default_conf из настроек
ConfField = Annotated[
    float | None,
    Form(ge=MODEL_CONF, le=1.0, description="Confidence threshold, defaults to default_conf from settings"),
]

# площадка с сохранённой сеткой; без неё сетка строится по этому же снимку
SiteField = Annotated[
    str | None,
    Form(pattern=r"^[A-Za-z0-9_-]{1,64}$", description="Calibrated site; omit to build the grid from this image"),
]

# формат выдачи отрисовки; в форме поле называется format
FormatField = Annotated[ImageFormat, Form(alias="format", description="Output image format")]

# большая сторона выдаваемого изображения, крупные кадры уменьшаются до неё
DEFAULT_RENDER_SIDE = 2000
MaxSideField = Annotated[int, Form(ge=64, le=8192, description="Max side of the output image in px")]


# разбор одного снимка: сетка мест, детекции на рабочем пороге и их сопоставление
@dataclass(frozen=True)
class Analysis:
    site_id: str | None
    slots: list[SlotModel]
    zone_polygon: np.ndarray
    status: str
    reason: str | None
    coverage: float
    boxes: np.ndarray
    assignment: Assignment
    summary: Summary
    inference_ms: int

    # машины в рабочей зоне: все детекции, кроме оказавшихся вне зоны
    @property
    def cars_in_zone(self) -> int:
        return len(self.boxes) - int(self.assignment.outside_zone.sum())


# общий для /detect и /render разбор: без site_id сетка строится по этому же снимку,
# с site_id берётся сохранённая; при failed-калибровке мест нет и остаётся только счёт машин
def analyse(
    image: np.ndarray, detector: CarDetector, settings: Settings, conf: float | None, site_id: str | None
) -> Analysis:
    threshold = settings.default_conf if conf is None else conf
    height, width = image.shape[:2]
    site = None
    if site_id is not None:
        site = load_site(settings, site_id)
        if (site.ref_width, site.ref_height) != (width, height):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Image is {width}x{height}, site {site_id} is calibrated for {site.ref_width}x{site.ref_height}",
            )

    start = time.perf_counter()
    if site is None:
        # детектор запускается один раз на меньшем из порогов: калибровке нужны боксы на default_conf
        # (7.2, шаг 1), занятости - на пороге запроса; фильтр по порогу в детекторе идёт после SAHI
        found, found_scores = detector.detect(image, min(threshold, settings.default_conf))
        calibration = calibrate(
            image, found[found_scores >= settings.default_conf], settings.calibration, settings.ground_offset
        )
        slots = slot_models(calibration)
        zone = calibration.markings.zone_polygon
        state = (calibration.status, calibration.reason, calibration.coverage)
        keep = found_scores >= threshold
        boxes, scores = found[keep], found_scores[keep]
    else:
        boxes, scores = detector.detect(image, threshold)
        slots = site.slots
        zone = np.array(site.zone_polygon, dtype=np.float32).reshape(-1, 2)
        state = (site.status, site.reason, site.calibration_coverage)

    polygons = [np.array(slot.polygon, dtype=np.float32) for slot in slots]
    assignment = assign_detections(polygons, boxes, scores, zone, settings.ground_offset)
    summary = summarize(assignment.occupied, np.array([slot.confirmed for slot in slots], dtype=bool))
    inference_ms = round((time.perf_counter() - start) * 1000)
    logger.info(
        "Occupancy on %dx%d image at conf %.3f in %d ms: %d cars, calibration %s %s, %d slots, %d occupied",
        width,
        height,
        threshold,
        inference_ms,
        len(boxes),
        state[0],
        state[1] or "",
        summary.total,
        summary.occupied,
    )
    return Analysis(site_id, slots, zone, *state, boxes, assignment, summary, inference_ms)


# ответ по разделу 7.3 из разбора снимка; общий для API и cli.py
def build_response(result: Analysis) -> OccupancyResponse:
    summary, assignment = result.summary, result.assignment
    rows: dict[int, list[int]] = {}
    for slot, busy in zip(result.slots, assignment.occupied):
        counts = rows.setdefault(slot.row, [0, 0])
        counts[0] += 1
        counts[1] += int(busy)
    return OccupancyResponse(
        site_id=result.site_id,
        timestamp=datetime.now().astimezone(),
        total_slots=summary.total,
        confirmed_slots=summary.confirmed,
        unconfirmed_slots=summary.unconfirmed,
        occupied=summary.occupied,
        free=summary.free,
        occupancy_rate=None if summary.rate is None else round(summary.rate, 3),
        cars_in_zone=result.cars_in_zone,
        unassigned_detections=int(assignment.unassigned.sum()),
        outside_zone_detections=int(assignment.outside_zone.sum()),
        calibration_status=result.status,
        calibration_reason=result.reason,
        calibration_coverage=round(result.coverage, 4),
        rows=[RowState(row=row, total=total, occupied=busy) for row, (total, busy) in sorted(rows.items())],
        slots=[
            SlotState(
                slot_id=slot.slot_id,
                occupied=bool(busy),
                confirmed=bool(slot.confirmed or busy),
                conf=round(float(score), 4) if busy else None,
                source=slot.source,
                polygon=slot.polygon,
            )
            for slot, busy, score in zip(result.slots, assignment.occupied, assignment.conf)
        ],
        inference_ms=result.inference_ms,
    )


# занятость мест по разделу 7.3; обработчики синхронные, FastAPI выполняет их в пуле потоков
@router.post("/detect")
def detect(
    file: ImageFile,
    detector: DetectorDep,
    settings: SettingsDep,
    conf: ConfField = None,
    site_id: SiteField = None,
) -> OccupancyResponse:
    return build_response(analyse(read_image(file, settings.max_upload_mb), detector, settings, conf, site_id))


# сводка для отрисовки; без мест - счёт машин в зоне и причина, по которой сетки нет
def summary_text(result: Analysis) -> str:
    summary = result.summary
    if not summary.total:
        return f"Cars in zone: {result.cars_in_zone} | calibration {result.status} {result.reason or ''}".rstrip()
    rate = "n/a" if summary.rate is None else f"{summary.rate:.0%}"
    return (
        f"Occupied {summary.occupied}/{summary.confirmed} ({rate}) | free {summary.free} | "
        f"unconfirmed {summary.unconfirmed} | no slot {int(result.assignment.unassigned.sum())}"
    )


# отрисовка разбора по разделу 7.4; общая для API и cli.py
def draw(image: np.ndarray, result: Analysis, max_side: int) -> np.ndarray:
    assignment = result.assignment
    # сначала уменьшаем кадр, потом рисуем: так толщина линий считается от итогового размера
    small, k = fit_max_side(image, max_side)
    return render_occupancy(
        small,
        [np.array(slot.polygon, dtype=np.float32) * k for slot in result.slots],
        assignment.occupied,
        np.array([slot.confirmed for slot in result.slots], dtype=bool),
        result.boxes[assignment.unassigned] * k,
        result.boxes[assignment.outside_zone] * k,
        result.zone_polygon * k,
        summary_text(result),
    )


# то же, но ответ - изображение с занятостью по разделу 7.4, по умолчанию JPEG не больше 2000 px
@router.post(
    "/render",
    response_class=Response,
    responses={status.HTTP_200_OK: {"content": {"image/jpeg": {}, "image/png": {}}}},
)
def render(
    file: ImageFile,
    detector: DetectorDep,
    settings: SettingsDep,
    conf: ConfField = None,
    site_id: SiteField = None,
    image_format: FormatField = "jpeg",
    max_side: MaxSideField = DEFAULT_RENDER_SIDE,
) -> Response:
    image = read_image(file, settings.max_upload_mb)
    canvas = draw(image, analyse(image, detector, settings, conf, site_id), max_side)
    return Response(content=encode_image(canvas, image_format), media_type=f"image/{image_format}")
