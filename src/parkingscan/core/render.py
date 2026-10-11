from typing import Literal

import cv2
import numpy as np

# форматы выдачи изображения и качество JPEG
ImageFormat = Literal["jpeg", "png"]
JPEG_QUALITY = 90

# цвета в RGB: занятые места - красные, свободные - зелёные, неподтверждённые места и детекции
# вне зоны - серые, детекции без места - жёлтые, контур зоны - голубой,
# сводка - белый текст на чёрной плашке
OCCUPIED_COLOR = (230, 40, 40)
FREE_COLOR = (40, 200, 60)
MUTED_COLOR = (170, 170, 170)
ZONE_COLOR = (0, 220, 255)
UNASSIGNED_COLOR = (255, 255, 0)
# непрозрачность заливки мест
FILL_ALPHA = 0.35
TEXT_COLOR = (255, 255, 255)
PLATE_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# толщина линий растёт с размером изображения, чтобы рамки читались и на 4000x3000
def line_thickness(image: np.ndarray) -> int:
    return max(2, max(image.shape[:2]) // 800)


# сводка в левом верхнем углу
def draw_summary(canvas: np.ndarray, text: str, thickness: int) -> None:
    scale = thickness * 0.5
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    pad = 4 * thickness
    cv2.rectangle(canvas, (0, 0), (width + 2 * pad, height + baseline + 2 * pad), PLATE_COLOR, -1)
    cv2.putText(canvas, text, (pad, pad + height), FONT, scale, TEXT_COLOR, thickness, cv2.LINE_AA)


# копия RGB-изображения с занятостью по разделу 7.4; координаты - в пикселях этого изображения;
# без мест (режим «только счёт») остаются зона, детекции и сводка
def render_occupancy(
    image: np.ndarray,
    polygons: list[np.ndarray],
    occupied: np.ndarray,
    confirmed: np.ndarray,
    unassigned_boxes: np.ndarray,
    outside_boxes: np.ndarray,
    zone_polygon: np.ndarray,
    summary: str,
) -> np.ndarray:
    canvas = image.copy()
    thickness = line_thickness(canvas)
    contours = [polygon.round().astype(np.int32) for polygon in polygons]
    colors = [OCCUPIED_COLOR if busy else FREE_COLOR for busy in occupied]

    # заливка только у занятых и свободных мест; неподтверждённое место не свободно и не занято,
    # поэтому у него только тонкий серый контур
    fill = canvas.copy()
    for contour, color, known in zip(contours, colors, occupied | confirmed):
        if known:
            cv2.fillPoly(fill, [contour], color)
    canvas = cv2.addWeighted(fill, FILL_ALPHA, canvas, 1 - FILL_ALPHA, 0)
    for contour, color, known in zip(contours, colors, occupied | confirmed):
        if known:
            cv2.polylines(canvas, [contour], True, color, thickness, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [contour], True, MUTED_COLOR, max(1, thickness // 2), cv2.LINE_AA)

    if len(zone_polygon) >= 3:
        cv2.polylines(canvas, [zone_polygon.round().astype(np.int32)], True, ZONE_COLOR, thickness, cv2.LINE_AA)
    for x1, y1, x2, y2 in outside_boxes.round().astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), MUTED_COLOR, max(1, thickness // 2))
    for x1, y1, x2, y2 in unassigned_boxes.round().astype(int):
        cv2.rectangle(canvas, (x1, y1), (x2, y2), UNASSIGNED_COLOR, thickness)
    draw_summary(canvas, summary, thickness)
    return canvas


# уменьшаем по большей стороне до max_side, меньшие изображения не увеличиваем
def fit_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    k = max_side / max(image.shape[:2])
    if k >= 1:
        return image, 1.0
    return cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), k


# кодируем RGB-изображение в JPEG или PNG
def encode_image(image: np.ndarray, image_format: ImageFormat) -> bytes:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image_format == "jpeg":
        ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError(f"{image_format} encoding failed")
    return buffer.tobytes()
