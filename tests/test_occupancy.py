import numpy as np
import pytest

from parkingscan.core.geometry import ground_points
from parkingscan.core.occupancy import Assignment, assign_detections, summarize

GROUND_OFFSET = 0.15

# два места с общей полосой x 100-110 и рабочая зона вокруг них
SLOT_A = np.array([[0, 0], [110, 0], [110, 100], [0, 100]], dtype=np.float32)
SLOT_B = np.array([[100, 0], [200, 0], [200, 100], [100, 100]], dtype=np.float32)
ZONE = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)


# бокс, опорная точка которого при GROUND_OFFSET лежит в (x, y)
def box_at(x: float, y: float) -> list[float]:
    return [x - 20, y - 85, x + 20, y + 15]


# сопоставление набора опорных точек с двумя местами
def assign(points: list[tuple[float, float]], zone: np.ndarray = ZONE) -> Assignment:
    boxes = np.array([box_at(x, y) for x, y in points], dtype=np.float32).reshape(-1, 4)
    scores = np.linspace(0.9, 0.5, len(points)).astype(np.float32)
    return assign_detections([SLOT_A, SLOT_B], boxes, scores, zone, GROUND_OFFSET)


# опорная точка - середина нижней стороны бокса, поднятая на ground_offset его высоты
def test_ground_point() -> None:
    points = ground_points(np.array([[10, 20, 50, 120]], dtype=np.float32), GROUND_OFFSET)
    assert points[0] == pytest.approx([30, 105])


# точка на стыке полигонов: детекция уходит одному месту - тому, где она глубже
def test_junction_goes_to_deeper_slot() -> None:
    result = assign([(104, 50)])
    assert result.slot_of.tolist() == [0]
    assert result.occupied.tolist() == [True, False]
    result = assign([(107, 50)])
    assert result.slot_of.tolist() == [1]
    assert result.occupied.tolist() == [False, True]


# две детекции в одном месте: место занято один раз, уверенность - наибольшая
def test_two_detections_in_one_slot() -> None:
    result = assign([(40, 50), (60, 50)])
    assert result.occupied.tolist() == [True, False]
    assert result.conf[0] == pytest.approx(0.9)
    assert result.conf[1] == 0


# детекция без места: в зоне - unassigned, вне зоны - outside_zone
def test_unassigned_and_outside_zone() -> None:
    result = assign([(150, 50), (250, 250), (400, 400)])
    assert result.slot_of.tolist() == [1, -1, -1]
    assert result.unassigned.tolist() == [False, True, False]
    assert result.outside_zone.tolist() == [False, False, True]


# зоны нет (отказ no_scale): весь кадр считается зоной
def test_empty_zone_is_whole_frame() -> None:
    result = assign([(400, 400)], np.zeros((0, 2), dtype=np.float32))
    assert result.unassigned.tolist() == [True]
    assert result.outside_zone.tolist() == [False]


# загрузка считается по подтверждённым местам: занятое неподтверждённое место становится
# подтверждённым, пустое неподтверждённое не входит ни в свободные, ни в знаменатель
def test_rate_uses_confirmed_slots_only() -> None:
    occupied = np.array([True, False, False, False])
    confirmed = np.array([False, True, True, False])
    summary = summarize(occupied, confirmed)
    assert (summary.total, summary.confirmed, summary.unconfirmed) == (4, 3, 1)
    assert (summary.occupied, summary.free) == (1, 2)
    assert summary.rate == pytest.approx(1 / 3)


# мест нет или ни одно не подтверждено: загрузка не определена
def test_rate_is_none_without_confirmed_slots() -> None:
    empty = np.zeros(0, dtype=bool)
    assert summarize(empty, empty).rate is None
    assert summarize(np.array([False]), np.array([False])).rate is None
