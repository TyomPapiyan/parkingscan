import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from parkingscan.config import get_settings, load_inference_config
from parkingscan.core.calibration import Calibration, calibrate
from parkingscan.core.detector import CarDetector, select_device
from parkingscan.core.geometry import contains

# Ручная визуальная проверка калибровки по разделу 7.2 на снимках data/samples: рабочая зона,
# маска краски, разделители, места с цветом по источнику, номера рядов и статус.
# Для снимков из slots_gt.json печатается recall: доля эталонных опорных точек машин,
# попавших внутрь какого-либо найденного места.
# Запуск из корня репозитория: uv run python tests/check_markings.py <папка для картинок>
# Имя без префикса test_, поэтому pytest этот скрипт не собирает: ему нужны веса и GPU

# корень репозитория, скрипт лежит в tests/
ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"

# цвета в BGR: зона - голубой контур, краска - пурпурная заливка, склеенные пары - зелёные,
# одиночные края - оранжевые, машины - серые
ZONE_COLOR = (255, 255, 0)
PAINT_COLOR = (255, 0, 255)
PAIRED_COLOR = (0, 255, 0)
SINGLE_COLOR = (0, 140, 255)
CAR_COLOR = (160, 160, 160)

# места по источнику: наблюдаемые - зелёные, интерполированные - жёлтые, спроецированные
# с машиной - пурпурные, спроецированные без машины - синие
SOURCE_COLORS = {
    "observed": (0, 200, 0),
    "interpolated": (0, 220, 255),
    "projected": (255, 0, 200),
    "extrapolated": (255, 120, 0),
}

# эталонные опорные точки машин - красные
GROUND_TRUTH_COLOR = (0, 0, 255)


# картинка для просмотра в разрешении work: затемнённый кадр, маска, зона, машины, сегменты, места
def draw(image_bgr: np.ndarray, calibration: Calibration, boxes: np.ndarray, truth: np.ndarray) -> np.ndarray:
    markings = calibration.markings
    factor = markings.work_factor
    small = cv2.resize(image_bgr, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    canvas = (small * 0.55).astype(np.uint8)
    canvas[markings.paint_mask > 0] = PAINT_COLOR
    for x1, y1, x2, y2 in (boxes * factor).astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), CAR_COLOR, 1)
    if len(markings.zone_polygon):
        cv2.polylines(canvas, [(markings.zone_polygon * factor).astype(np.int32)], True, ZONE_COLOR, 3)

    # места: полупрозрачная заливка по источнику и контур
    fill = canvas.copy()
    for slot in calibration.slots:
        cv2.fillPoly(fill, [(slot.polygon * factor).astype(np.int32)], SOURCE_COLORS[slot.source])
    canvas = cv2.addWeighted(fill, 0.35, canvas, 0.65, 0)
    for slot in calibration.slots:
        cv2.polylines(canvas, [(slot.polygon * factor).astype(np.int32)], True, SOURCE_COLORS[slot.source], 2)

    for (x1, y1, x2, y2), paired in zip(markings.segments * factor, markings.paired):
        color, thickness = (PAIRED_COLOR, 3) if paired else (SINGLE_COLOR, 2)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)

    # номер ряда у его первого места
    for row in calibration.rows:
        first = next(s for s in calibration.slots if s.row == row.index)
        x, y = (first.polygon.mean(0) * factor).astype(int)
        cv2.putText(canvas, f"R{row.index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 8, cv2.LINE_AA)
        cv2.putText(canvas, f"R{row.index}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)

    for x, y in (truth * factor).astype(int):
        cv2.circle(canvas, (x, y), 9, GROUND_TRUTH_COLOR, -1)

    status = f"{calibration.status} {calibration.reason or ''} | slots {len(calibration.slots)}"
    status += f" | rows {len(calibration.rows)} | coverage {calibration.coverage:.2f}"
    cv2.rectangle(canvas, (0, 0), (1150, 60), (0, 0, 0), -1)
    cv2.putText(canvas, status, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    return canvas


# калибровка всех снимков data/samples, сводка в stdout и картинки в out_dir
def main() -> None:
    parser = argparse.ArgumentParser(description="Draw calibration results on data/samples images")
    parser.add_argument("out_dir", type=Path, help="folder for the output images")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    config = load_inference_config(ROOT / "models" / "config.json")
    detector = CarDetector(ROOT / "models" / config.model_file, config, select_device()[0])
    detector.warmup()
    names = list(json.loads((SAMPLES_DIR / "ground_truth.json").read_text(encoding="utf-8")))
    slots_truth = json.loads((SAMPLES_DIR / "slots_gt.json").read_text(encoding="utf-8"))

    for name in names:
        image_bgr = cv2.imread(str(SAMPLES_DIR / name))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(image, settings.default_conf)
        start = time.perf_counter()
        calibration = calibrate(image, boxes, settings.calibration, settings.ground_offset)
        ms = (time.perf_counter() - start) * 1000
        truth = np.array([s["ground"] for s in slots_truth.get(name, [])], dtype=np.float32).reshape(-1, 2)
        line = (
            f"{name}: {calibration.status} {calibration.reason or ''} | cars {len(boxes)} | "
            f"segments {len(calibration.markings.segments)} | rows {len(calibration.rows)} | "
            f"slots {len(calibration.slots)} | coverage {calibration.coverage:.2f} | {ms:.0f} ms"
        )
        if len(truth):
            found = sum(any(contains(s.polygon, point) for s in calibration.slots) for point in truth)
            line += f" | recall {found}/{len(truth)} = {found / len(truth):.2f}"
        print(line)
        cv2.imwrite(str(args.out_dir / f"{Path(name).stem}_slots.jpg"), draw(image_bgr, calibration, boxes, truth))


if __name__ == "__main__":
    main()
