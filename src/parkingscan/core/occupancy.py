from dataclasses import dataclass

import numpy as np

from parkingscan.core.geometry import ground_points, signed_depth


# сопоставление детекций с местами по разделу 7.3
@dataclass(frozen=True)
class Assignment:
    # по местам: занято ли место и наибольшая уверенность детекции в нём, у пустого - 0
    occupied: np.ndarray
    conf: np.ndarray
    # по детекциям: номер места или -1, признаки «в зоне без места» и «вне зоны»
    slot_of: np.ndarray
    unassigned: np.ndarray
    outside_zone: np.ndarray


# счётчики мест; загрузка считается только по подтверждённым местам
@dataclass(frozen=True)
class Summary:
    total: int
    confirmed: int
    unconfirmed: int
    occupied: int
    free: int
    rate: float | None


# каждая детекция назначается не более чем одному месту - тому, внутри которого её опорная точка
# лежит глубже всего; сначала место, потом зона: точка внутри места занимает его, даже если
# упрощённый контур зоны её срезал; зона без контура (отказ no_scale) - весь кадр
def assign_detections(
    polygons: list[np.ndarray],
    boxes: np.ndarray,
    scores: np.ndarray,
    zone_polygon: np.ndarray,
    ground_offset: float,
) -> Assignment:
    points = ground_points(boxes, ground_offset)
    slot_of = np.full(len(boxes), -1)
    in_zone = np.ones(len(boxes), dtype=bool)
    occupied = np.zeros(len(polygons), dtype=bool)
    conf = np.zeros(len(polygons), dtype=np.float32)
    for index, point in enumerate(points):
        depths = [signed_depth(polygon, point) for polygon in polygons]
        if depths and max(depths) >= 0:
            slot = int(np.argmax(depths))
            slot_of[index] = slot
            occupied[slot] = True
            conf[slot] = max(conf[slot], scores[index])
        elif len(zone_polygon) >= 3:
            in_zone[index] = signed_depth(zone_polygon, point) >= 0
    free_standing = slot_of < 0
    return Assignment(occupied, conf, slot_of, free_standing & in_zone, free_standing & ~in_zone)


# место подтверждено калибровкой либо занято сейчас; неподтверждённое пустое место не входит
# ни в свободные, ни в знаменатель загрузки, иначе оно раздувает число свободных мест
def summarize(occupied: np.ndarray, confirmed: np.ndarray) -> Summary:
    confirmed_now = int((confirmed | occupied).sum())
    busy = int(occupied.sum())
    return Summary(
        total=len(occupied),
        confirmed=confirmed_now,
        unconfirmed=len(occupied) - confirmed_now,
        occupied=busy,
        free=confirmed_now - busy,
        rate=busy / confirmed_now if confirmed_now else None,
    )
