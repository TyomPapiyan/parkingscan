import cv2
import numpy as np


# опорные точки боксов по разделу 7.3: середина нижней стороны, поднятая на ground_offset высоты бокса;
# на косом ракурсе кузов проецируется вверх по кадру, и только низ бокса лежит на своём месте
def ground_points(boxes: np.ndarray, ground_offset: float) -> np.ndarray:
    x = (boxes[:, 0] + boxes[:, 2]) / 2
    y = boxes[:, 3] - ground_offset * (boxes[:, 3] - boxes[:, 1])
    return np.stack([x, y], 1).astype(np.float32)


# расстояние со знаком от точки до границы полигона: положительное внутри, отрицательное снаружи
def signed_depth(polygon: np.ndarray, point: np.ndarray) -> float:
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True)


# лежит ли точка внутри полигона или на его границе
def contains(polygon: np.ndarray, point: np.ndarray) -> bool:
    return signed_depth(polygon, point) >= 0


# IoU двух выпуклых четырёхугольников
def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    inter, _ = cv2.intersectConvexConvex(a32, b32)
    union = cv2.contourArea(a32) + cv2.contourArea(b32) - inter
    return float(inter / union) if union > 0 else 0.0
