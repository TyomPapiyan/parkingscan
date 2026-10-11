import cv2
import numpy as np
import pytest

from parkingscan.config import CalibrationSettings
from parkingscan.core.calibration import Calibration, calibrate
from parkingscan.core.geometry import contains, ground_points

# синтетический кадр по разделу 13: ряд белых разделителей на сером асфальте с шумом, зебра,
# «машины» поверх двух разделителей, регулярные тонкие линии без машин вдали от ряда;
# машина 200x450 px, шаг ряда 225 px, разделитель 450 px, линия краски 12 px
FRAME_SHAPE = (3000, 4000, 3)
CAR_W, CAR_H = 200, 450
STEP = 225
ROW_X0, ROW_TOP, ROW_BOTTOM = 400, 1800, 2250
ROW_COUNT = 8
PAINT = (235, 235, 235)
ASPHALT = 90
GROUND_OFFSET = 0.15

# машины поверх разделителей 3 и 6: бокс сдвинут так, что опорная точка лежит внутри места слева
HIDDEN = (3, 6)

# машины для модели масштаба, ниже ряда и вдали от разметки
SCALE_CARS_Y = 2450
SCALE_CARS_X = (300, 700, 1100, 1500, 1900)


# ровный асфальт с мелким шумом
def asphalt(rng: np.random.Generator) -> np.ndarray:
    noise = rng.integers(0, 12, FRAME_SHAPE[:2] + (1,), dtype=np.uint8)
    return np.full(FRAME_SHAPE, ASPHALT, dtype=np.uint8) + noise


# боксы машин [x1, y1, x2, y2]: две на ряду и пять для масштаба
def car_boxes() -> np.ndarray:
    boxes = []
    for index in HIDDEN:
        x = ROW_X0 + index * STEP
        boxes.append([x - 150, ROW_TOP + 20, x + 50, ROW_TOP + 20 + CAR_H])
    for x in SCALE_CARS_X:
        boxes.append([x, SCALE_CARS_Y, x + CAR_W, SCALE_CARS_Y + CAR_H])
    return np.array(boxes, dtype=np.float32)


# полный синтетический кадр и боксы машин
def synthetic_lot() -> tuple[np.ndarray, np.ndarray]:
    image = asphalt(np.random.default_rng(0))
    for index in range(ROW_COUNT):
        x = ROW_X0 + index * STEP
        cv2.line(image, (x, ROW_TOP), (x, ROW_BOTTOM), PAINT, 12)
    # зебра: широкие полосы, толще любого разделителя
    for index in range(4):
        y = 1850 + index * 160
        cv2.rectangle(image, (2600, y), (3500, y + 80), PAINT, -1)
    # «фасад»: регулярные тонкие линии без машин, дальше row_neighbour_gap от ряда
    for index in range(6):
        x = 2500 + index * STEP
        cv2.line(image, (x, 300), (x, 750), PAINT, 12)
    boxes = car_boxes()
    # машины - тёмные прямоугольники, закрывающие свои разделители
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 40, 45), -1)
    return image, boxes


@pytest.fixture(scope="module")
def result() -> Calibration:
    image, boxes = synthetic_lot()
    return calibrate(image, boxes, CalibrationSettings(), GROUND_OFFSET)


# ряд восстановлен целиком: 8 разделителей, из них 2 интерполированы, 7 мест
def test_row_is_recovered_with_interpolation(result: Calibration) -> None:
    assert len(result.rows) == 1
    assert result.rows[0].sources.count("interpolated") == len(HIDDEN)
    assert len(result.rows[0].sources) == ROW_COUNT
    assert len(result.slots) == ROW_COUNT - 1


# места рядом со скрытыми разделителями - интерполированные, остальные - наблюдаемые
def test_slot_sources(result: Calibration) -> None:
    sources = [s.source for s in sorted(result.slots, key=lambda s: s.index)]
    expected = ["observed", "observed", "interpolated", "interpolated", "observed", "interpolated", "interpolated"]
    assert sources == expected
    # подтверждены три наблюдаемых места и два интерполированных, в которых стоят машины
    assert result.coverage == pytest.approx(5 / 7)
    assert result.status == "ok"
    assert result.reason is None


# ни зебра, ни регулярные линии без машин не дают мест
def test_zebra_and_facade_give_no_slots(result: Calibration) -> None:
    for slot in result.slots:
        center = slot.polygon.mean(0)
        assert center[0] < 2400
        assert center[1] > 1500


# гибрид 7.2: ряд из 7 мест, видны только разделители 0-2, остальные закрыты машинами -
# боксы машин в местах 3, 4 и 6 с расширением car_mask_dilate вырезают разделители 3-7 из маски;
# места 2 и 5 свободны
COVERED_SLOTS = 7
COVERED_CARS = (3, 4, 6)


# боксы машин [x1, y1, x2, y2] по центру своих мест и пять машин для масштаба
def covered_boxes() -> np.ndarray:
    boxes = []
    for index in COVERED_CARS:
        x = ROW_X0 + index * STEP + (STEP - CAR_W) // 2
        boxes.append([x, ROW_TOP + 20, x + CAR_W, ROW_TOP + 20 + CAR_H])
    for x in SCALE_CARS_X:
        boxes.append([x, SCALE_CARS_Y, x + CAR_W, SCALE_CARS_Y + CAR_H])
    return np.array(boxes, dtype=np.float32)


@pytest.fixture(scope="module")
def covered() -> Calibration:
    image = asphalt(np.random.default_rng(3))
    for index in range(COVERED_SLOTS + 1):
        x = ROW_X0 + index * STEP
        cv2.line(image, (x, ROW_TOP), (x, ROW_BOTTOM), PAINT, 12)
    boxes = covered_boxes()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 40, 45), -1)
    return calibrate(image, boxes, CalibrationSettings(), GROUND_OFFSET)


# ряд продолжен под машинами ходом по сетке: источник места - по качеству границ и по машине,
# за последней машиной ход останавливается
def test_row_continues_under_cars(covered: Calibration) -> None:
    assert len(covered.rows) == 1
    assert covered.rows[0].sources == ["observed"] * 3 + ["projected"] * 5
    sources = [s.source for s in sorted(covered.slots, key=lambda s: s.index)]
    expected = ["observed", "observed", "extrapolated", "projected", "projected", "extrapolated", "projected"]
    assert sources == expected
    # не подтверждены только два свободных спроецированных места
    assert covered.coverage == pytest.approx(5 / 7)
    assert covered.status == "ok"


# опорная точка каждой машины лежит в своём месте
def test_projected_slots_hold_their_cars(covered: Calibration) -> None:
    slots = sorted(covered.slots, key=lambda s: s.index)
    points = ground_points(covered_boxes()[: len(COVERED_CARS)], GROUND_OFFSET)
    for index, point in zip(COVERED_CARS, points):
        assert contains(slots[index].polygon, point)


# цепочка ровно из двух разделителей с шагом в два места отбрасывается: у пары нельзя
# отличить одно место от двух
def test_wide_pair_is_rejected() -> None:
    image = asphalt(np.random.default_rng(4))
    for x in (ROW_X0, ROW_X0 + 2 * STEP):
        cv2.line(image, (x, ROW_TOP), (x, ROW_BOTTOM), PAINT, 12)
    # машина между разделителями, вдали от обоих, чтобы маска машин их не вырезала
    car = [ROW_X0 + STEP - CAR_W // 2, ROW_TOP + 20, ROW_X0 + STEP + CAR_W // 2, ROW_TOP + 20 + CAR_H]
    boxes = np.concatenate([np.array([car], dtype=np.float32), covered_boxes()[len(COVERED_CARS) :]])
    result = calibrate(image, boxes, CalibrationSettings(), GROUND_OFFSET)
    assert result.status == "failed"
    assert result.reason == "no_rows"


# пустой кадр без машин: отказ с причиной, без исключения
def test_empty_frame_fails_without_scale() -> None:
    image = asphalt(np.random.default_rng(1))
    result = calibrate(image, np.zeros((0, 4), dtype=np.float32), CalibrationSettings(), GROUND_OFFSET)
    assert result.status == "failed"
    assert result.reason == "no_scale"
    assert result.slots == []


# машины есть, разметки нет: отказ no_markings
def test_frame_without_markings_fails() -> None:
    image = asphalt(np.random.default_rng(2))
    result = calibrate(image, car_boxes(), CalibrationSettings(), GROUND_OFFSET)
    assert result.status == "failed"
    assert result.reason == "no_markings"
