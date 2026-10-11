import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi import Path as PathParam

from parkingscan.api.deps import DetectorDep, SettingsDep, read_image
from parkingscan.api.schemas import RowModel, SiteCalibration, SlotModel, SlotsUpdate
from parkingscan.config import Settings
from parkingscan.core.calibration import Calibration, calibrate, merge_calibrations

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sites", tags=["sites"])

# идентификатор площадки становится именем файла, поэтому только безопасные символы
SiteId = Annotated[str, PathParam(pattern=r"^[A-Za-z0-9_-]{1,64}$", description="Site identifier")]

# снимки площадки с одной позиции камеры
ImageFiles = Annotated[list[UploadFile], File(description="One or more images from the same camera position")]


# файл калибровки площадки
def site_file(settings: Settings, site_id: str) -> Path:
    return settings.data_dir / "sites" / f"{site_id}.json"


# калибровка сохраняется целиком, каталог создаётся при первой записи
def save_site(settings: Settings, site: SiteCalibration) -> None:
    path = site_file(settings, site.site_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(site.model_dump_json(indent=2), encoding="utf-8")


# сохранённая калибровка или 404
def load_site(settings: Settings, site_id: str) -> SiteCalibration:
    path = site_file(settings, site_id)
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Site {site_id} is not calibrated")
    return SiteCalibration.model_validate_json(path.read_text(encoding="utf-8"))


# места калибровки в формате API и файла
def slot_models(calibration: Calibration) -> list[SlotModel]:
    return [
        SlotModel(
            slot_id=f"r{slot.row}_s{slot.index:02d}",
            row=slot.row,
            index=slot.index,
            polygon=slot.polygon.round(1).tolist(),
            confidence=round(slot.confidence, 3),
            source=slot.source,
            confirmed=slot.confirmed,
        )
        for slot in calibration.slots
    ]


# результат калибровки в формате API и файла; size - высота и ширина кадра
def to_site(
    site_id: str, calibration: Calibration, size: tuple[int, int], images: int, settings: Settings
) -> SiteCalibration:
    height, width = size
    return SiteCalibration(
        site_id=site_id,
        ref_width=width,
        ref_height=height,
        images=images,
        calibrated_at=datetime.now().astimezone(),
        status=calibration.status,
        reason=calibration.reason,
        calibration_coverage=round(calibration.coverage, 4),
        zone_polygon=calibration.markings.zone_polygon.round(1).tolist(),
        params=settings.calibration.model_dump(),
        rows=[
            RowModel(row=row.index, dividers=row.dividers.round(1).tolist(), sources=row.sources)
            for row in calibration.rows
        ],
        slots=slot_models(calibration),
    )


# построение сетки мест по разметке (7.2); синхронно: один кадр калибруется за 1-2 с,
# а клиенту сразу нужен статус и причина отказа
@router.post("/{site_id}/calibrate")
def calibrate_site(
    site_id: SiteId, files: ImageFiles, detector: DetectorDep, settings: SettingsDep
) -> SiteCalibration:
    images = [read_image(upload, settings.max_upload_mb) for upload in files]
    sizes = {image.shape[:2] for image in images}
    if len(sizes) > 1:
        raise HTTPException(status.HTTP_409_CONFLICT, "Images differ in size, expected one camera position")
    start = time.perf_counter()
    results = []
    for image in images:
        boxes, _ = detector.detect(image, settings.default_conf)
        results.append(calibrate(image, boxes, settings.calibration, settings.ground_offset))
    calibration = merge_calibrations(results, settings.calibration)
    logger.info(
        "Site %s calibrated from %d image(s) in %d ms: %s %s, %d slots, coverage %.2f",
        site_id,
        len(images),
        round((time.perf_counter() - start) * 1000),
        calibration.status,
        calibration.reason or "",
        len(calibration.slots),
        calibration.coverage,
    )
    if calibration.status == "failed":
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"Calibration failed: {calibration.reason}")
    site = to_site(site_id, calibration, sizes.pop(), len(images), settings)
    save_site(settings, site)
    return site


# текущая сетка площадки
@router.get("/{site_id}/slots")
def get_slots(site_id: SiteId, settings: SettingsDep) -> SiteCalibration:
    return load_site(settings, site_id)


# ручная правка: сетка заменяется целиком, места получают source manual, confidence 1 и подтверждены;
# разделители рядов после правки неизвестны, поэтому ряды остаются только номерами у мест
@router.put("/{site_id}/slots")
def put_slots(site_id: SiteId, update: SlotsUpdate, settings: SettingsDep) -> SiteCalibration:
    site = load_site(settings, site_id)
    slots = [
        SlotModel(
            slot_id=f"r{slot.row}_s{slot.index:02d}",
            row=slot.row,
            index=slot.index,
            polygon=slot.polygon,
            confidence=1.0,
            source="manual",
            confirmed=True,
        )
        for slot in update.slots
    ]
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Duplicate row and index pairs")
    site = site.model_copy(
        update={
            "slots": slots,
            "rows": [],
            "status": "manual",
            "reason": None,
            "calibration_coverage": 1.0,
            "calibrated_at": datetime.now().astimezone(),
        }
    )
    save_site(settings, site)
    return site
