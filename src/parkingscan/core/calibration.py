from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from sklearn.linear_model import HuberRegressor

from parkingscan.config import CalibrationSettings
from parkingscan.core.geometry import contains, ground_points, polygon_iou

# допустимое отклонение пропуска от целого числа медианных шагов при заполнении интерполяцией
FILL_TOLERANCE = 0.3

# уверенность места по источникам границ, раздел 7.2 шаг 9
CONFIDENCE_ONE_INTERPOLATED = 0.6
CONFIDENCE_PROJECTED = 0.7
CONFIDENCE_WEAK = 0.4

# ширина ячейки хода после подтяжки к опорной точке машины, в шагах ряда
PULL_RANGE = (0.7, 1.3)

# покрытие, начиная с которого калибровка считается полной
COVERAGE_OK = 0.6

# места с разных снимков одной позиции считаются одним местом при IoU не ниже этого значения
MERGE_IOU = 0.5

# конец разделителя ближе этой доли большей стороны кадра к краю считается обрезанным кадром
FRAME_MARGIN = 0.01

# соседние размеры ядер top-hat отличаются в эту кратность: ядро подбирается полосами по масштабу,
# а не для каждого пикселя
KERNEL_STEP = 1.25

# жёлтая краска по HSV в шкале OpenCV (H 0-179): диапазон тона и минимальная насыщенность
YELLOW_HUE = (15, 40)
YELLOW_MIN_SATURATION = 80


# модель локального масштаба: логарифм короткой и длинной стороны бокса машины линейно по координатам;
# по логарифму, потому что размер машины внутри кадра меняется до 25 раз
@dataclass(frozen=True)
class ScaleModel:
    short_coef: np.ndarray
    long_coef: np.ndarray
    # координаты нормируются на большую сторону кадра, чтобы регуляризация Huber не зависела от разрешения
    norm: float
    # предсказание ограничивается наблюдаемым диапазоном размеров боксов: за краем облака машин
    # экспонента уходит в тысячи пикселей и ломает все пороги в долях масштаба
    short_range: tuple[float, float]
    long_range: tuple[float, float]

    # ожидаемая ширина машины в пикселях исходного кадра в точках (x, y)
    def short_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = self.short_coef
        return np.clip(np.exp(c[0] + c[1] * x / self.norm + c[2] * y / self.norm), *self.short_range)

    # ожидаемая длинная сторона бокса машины в пикселях исходного кадра в точках (x, y)
    def long_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        c = self.long_coef
        return np.clip(np.exp(c[0] + c[1] * x / self.norm + c[2] * y / self.norm), *self.long_range)


# результат шагов 1-5: модель масштаба, рабочая зона, маска краски и сегменты разметки;
# координаты сегментов и зоны - в пикселях исходного кадра, маски - в разрешении work_side
@dataclass(frozen=True)
class Markings:
    status: Literal["ok", "failed"]
    reason: str | None
    scale: ScaleModel | None
    work_factor: float
    zone_mask: np.ndarray
    zone_polygon: np.ndarray
    paint_mask: np.ndarray
    segments: np.ndarray
    paired: np.ndarray


# робастная регрессия log(стороны бокса) по нормированным координатам центра бокса
def fit_scale_model(boxes: np.ndarray, image_side: int, min_cars: int) -> ScaleModel | None:
    if len(boxes) < min_cars:
        return None
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1) / image_side
    short, long = np.minimum(width, height), np.maximum(width, height)
    coefs = []
    for side in (short, long):
        huber = HuberRegressor().fit(centers, np.log(side))
        coefs.append(np.array([huber.intercept_, *huber.coef_]))
    return ScaleModel(
        short_coef=coefs[0],
        long_coef=coefs[1],
        norm=float(image_side),
        short_range=(float(short.min()), float(short.max())),
        long_range=(float(long.min()), float(long.max())),
    )


# значение карты масштаба в серединах сегментов
def width_at(width_map: np.ndarray, segments: np.ndarray) -> np.ndarray:
    mid = ((segments[:, :2] + segments[:, 2:]) / 2).astype(int)
    rows = mid[:, 1].clip(0, width_map.shape[0] - 1)
    cols = mid[:, 0].clip(0, width_map.shape[1] - 1)
    return width_map[rows, cols]


# длина сегментов кадра work в долях локальной длинной стороны машины
def relative_length(segments: np.ndarray, scale: ScaleModel, factor: float) -> np.ndarray:
    mid = (segments[:, :2] + segments[:, 2:]) / 2 / factor
    length = np.linalg.norm(segments[:, 2:] - segments[:, :2], axis=1) / factor
    return length / scale.long_at(mid[:, 0], mid[:, 1])


# рабочая зона в разрешении work: машина достаточно крупная и точка не на периферии фишая
def working_zone(width_map: np.ndarray, factor: float, params: CalibrationSettings) -> np.ndarray:
    height, width = width_map.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.hypot(xs - width / 2, ys - height / 2) / (np.hypot(width, height) / 2)
    zone = (width_map / factor >= params.roi_min_car_px) & (radius <= params.roi_max_radius)
    return zone.astype(np.uint8) * 255


# контур наибольшей связной области зоны, в пикселях исходного кадра
def zone_outline(zone: np.ndarray, factor: float) -> np.ndarray:
    contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    largest = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(largest, 0.002 * cv2.arcLength(largest, True), True)
    return polygon.reshape(-1, 2).astype(np.float32) / factor


# маска краски: white top-hat по каналу L с ядром по локальному масштабу, порог Оцу внутри зоны,
# вычитание машин и отсев толстых штрихов
def paint_mask(
    image: np.ndarray,
    zone: np.ndarray,
    width_map: np.ndarray,
    car_boxes: np.ndarray,
    params: CalibrationSettings,
) -> np.ndarray:
    lightness = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 0]

    # белые машины дают яркие блики - верхняя часть бокса, где кузов и крыша, расширенная на долю
    # размера бокса, исключается до порога, иначе крыши завышают порог и слабая разметка теряется;
    # нижняя полоса бокса остаётся: разделители лежат у нижней кромки, а низ машины обычно в тени
    cars = np.zeros_like(zone)
    for x1, y1, x2, y2 in car_boxes:
        pad = params.car_mask_dilate * max(x2 - x1, y2 - y1)
        bottom = y2 - params.car_mask_keep_bottom * (y2 - y1)
        cv2.rectangle(cars, (int(x1 - pad), int(y1 - pad)), (int(x2 + pad), int(bottom)), 255, -1)
    inside = (zone > 0) & (cars == 0)
    if not inside.any():
        return np.zeros_like(zone)

    # ядра квантуются геометрически; каждый пиксель берёт top-hat своей полосы масштаба
    # и порог из своей окрестности: среднее top-hat плюс paint_sigma стандартных отклонений
    # в окне размером в ширину машины; на пёстрой плитке и листве порог сам поднимается,
    # на гладком покрытии с линией остаётся низким
    kernel_map = np.maximum(params.tophat_kernel * width_map, 3.0)
    levels = np.round(np.log(kernel_map) / np.log(KERNEL_STEP)).astype(int)
    tophat = np.zeros(lightness.shape, dtype=np.float32)
    threshold = np.zeros(lightness.shape, dtype=np.float32)
    for level in np.unique(levels[inside]):
        size = int(round(KERNEL_STEP**level)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        band = inside & (levels == level)
        response = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        window = (int(size / params.tophat_kernel), int(size / params.tophat_kernel))
        mean = cv2.blur(response, window)
        std = np.sqrt(np.maximum(cv2.blur(response * response, window) - mean * mean, 0))
        tophat[band] = response[band]
        threshold[band] = (mean + params.paint_sigma * std)[band]
    bright = tophat >= np.maximum(threshold, params.paint_min_contrast)

    # краска белая или жёлтая: цветные яркие пятна - листва, цветы, вывески - краской не считаются
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation = hsv[:, :, 0], hsv[:, :, 1]
    white = saturation <= params.paint_max_saturation
    yellow = (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1]) & (saturation >= YELLOW_MIN_SATURATION)
    mask = (bright & (white | yellow) & inside).astype(np.uint8) * 255

    # толщина штриха компоненты - удвоенный максимум distance transform внутри неё;
    # зебра, штриховка, стрелки и цифры толще разделителей
    count, labels = cv2.connectedComponents(mask, connectivity=8)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    stroke = np.zeros(count, dtype=np.float32)
    np.maximum.at(stroke, labels.ravel(), distance.ravel())
    pixels = np.bincount(labels.ravel(), minlength=count)
    local_width = np.bincount(labels.ravel(), weights=width_map.ravel(), minlength=count) / np.maximum(pixels, 1)
    too_thick = 2 * stroke > params.max_stroke * local_width
    too_thick[0] = False
    mask[too_thick[labels]] = 0
    return mask


# склейка парных краёв: LSD даёт по сегменту на каждый край полосы краски,
# пара почти параллельных близких перекрывающихся сегментов заменяется осевой линией
def merge_edge_pairs(
    segments: np.ndarray, width_at_mid: np.ndarray, params: CalibrationSettings
) -> tuple[np.ndarray, np.ndarray]:
    count = len(segments)
    if count == 0:
        return segments, np.zeros(0, dtype=bool)
    p1, p2 = segments[:, :2], segments[:, 2:]
    mid = (p1 + p2) / 2
    vec = p2 - p1
    length = np.linalg.norm(vec, axis=1)
    direction = vec / np.maximum(length, 1e-6)[:, None]
    normal = np.stack([-direction[:, 1], direction[:, 0]], 1)

    # угол между прямыми без учёта направления сегмента
    angle_ok = np.abs(direction @ direction.T) >= np.cos(np.radians(params.merge_angle_deg))
    # расстояние между прямыми: среднее от середины одного сегмента до прямой другого
    offset = mid[None, :, :] - mid[:, None, :]
    dist = (np.abs(np.einsum("ijk,ik->ij", offset, normal)) + np.abs(np.einsum("ijk,jk->ij", offset, normal))) / 2
    dist_limit = params.merge_dist * np.minimum.outer(width_at_mid, width_at_mid)
    # перекрытие проекций вдоль направления первого сегмента не меньше половины короткого
    t1 = np.einsum("ijk,ik->ij", p1[None, :, :] - p1[:, None, :], direction)
    t2 = np.einsum("ijk,ik->ij", p2[None, :, :] - p1[:, None, :], direction)
    overlap = np.minimum(np.maximum(t1, t2), length[:, None]) - np.maximum(np.minimum(t1, t2), 0)
    overlap_ok = overlap >= 0.5 * np.minimum.outer(length, length)

    candidate = angle_ok & (dist <= dist_limit) & overlap_ok
    np.fill_diagonal(candidate, False)
    pairs = np.argwhere(np.triu(candidate))
    order = np.argsort(dist[pairs[:, 0], pairs[:, 1]])

    used = np.zeros(count, dtype=bool)
    merged, paired = [], []
    for i, j in pairs[order]:
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        # осевая линия: общее направление, середина между сегментами, концы - объединение проекций
        d = direction[i] + (direction[j] if direction[i] @ direction[j] >= 0 else -direction[j])
        d /= np.linalg.norm(d)
        center = (mid[i] + mid[j]) / 2
        t = (np.array([p1[i], p2[i], p1[j], p2[j]]) - center) @ d
        merged.append([*(center + t.min() * d), *(center + t.max() * d)])
        paired.append(True)
    for i in np.flatnonzero(~used):
        merged.append(segments[i].tolist())
        paired.append(False)
    return np.array(merged, dtype=np.float32).reshape(-1, 4), np.array(paired, dtype=bool)


# сшивка обрывков одной линии: LSD рвёт разделитель на кривизне фишая, на неровном крае маски
# и на вырезах под машины; почти коллинеарные сегменты с малым смещением и разрывом объединяются
def link_collinear(
    segments: np.ndarray, paired: np.ndarray, width_at_mid: np.ndarray, params: CalibrationSettings
) -> tuple[np.ndarray, np.ndarray]:
    count = len(segments)
    if count < 2:
        return segments, paired
    p1, p2 = segments[:, :2], segments[:, 2:]
    mid = (p1 + p2) / 2
    length = np.linalg.norm(p2 - p1, axis=1)
    direction = (p2 - p1) / np.maximum(length, 1e-6)[:, None]
    normal = np.stack([-direction[:, 1], direction[:, 0]], 1)

    angle_ok = np.abs(direction @ direction.T) >= np.cos(np.radians(params.link_angle_deg))
    offset = mid[None, :, :] - mid[:, None, :]
    lateral = (np.abs(np.einsum("ijk,ik->ij", offset, normal)) + np.abs(np.einsum("ijk,jk->ij", offset, normal))) / 2
    scale = np.minimum.outer(width_at_mid, width_at_mid)
    # разрыв между проекциями сегментов на направление первого; перекрытие даёт отрицательный разрыв
    t1 = np.einsum("ijk,ik->ij", p1[None, :, :] - p1[:, None, :], direction)
    t2 = np.einsum("ijk,ik->ij", p2[None, :, :] - p1[:, None, :], direction)
    gap = np.maximum(np.minimum(t1, t2) - length[:, None], -np.maximum(t1, t2))
    linked = angle_ok & (lateral <= params.merge_dist * scale) & (gap <= params.link_gap * scale)
    np.fill_diagonal(linked, False)

    # компоненты связности; каждая заменяется отрезком вдоль главной оси своих концов
    parent = np.arange(count)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in np.argwhere(np.triu(linked)):
        parent[root(i)] = root(j)
    roots = np.array([root(i) for i in range(count)])
    merged, flags = [], []
    for r in np.unique(roots):
        members = np.flatnonzero(roots == r)
        ends = np.concatenate([p1[members], p2[members]])
        center = ends.mean(0)
        axis = np.linalg.svd(ends - center)[2][0]
        t = (ends - center) @ axis
        merged.append([*(center + t.min() * axis), *(center + t.max() * axis)])
        flags.append(bool(paired[members].any()))
    return np.array(merged, dtype=np.float32).reshape(-1, 4), np.array(flags, dtype=bool)


# шаги 1-5 раздела 7.2 на одном RGB-снимке по уже найденным боксам машин
def find_markings(image: np.ndarray, car_boxes: np.ndarray, params: CalibrationSettings) -> Markings:
    height, width = image.shape[:2]
    factor = min(1.0, params.work_side / max(height, width))
    small = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else image
    empty_mask = np.zeros(small.shape[:2], dtype=np.uint8)

    scale = fit_scale_model(car_boxes, max(height, width), params.scale_min_cars)
    if scale is None:
        return Markings(
            status="failed",
            reason="no_scale",
            scale=None,
            work_factor=factor,
            zone_mask=empty_mask,
            zone_polygon=np.zeros((0, 2), dtype=np.float32),
            paint_mask=empty_mask,
            segments=np.zeros((0, 4), dtype=np.float32),
            paired=np.zeros(0, dtype=bool),
        )

    # ожидаемая ширина машины в каждом пикселе кадра work, в пикселях work
    ys, xs = np.mgrid[0 : small.shape[0], 0 : small.shape[1]].astype(np.float32)
    width_map = (scale.short_at(xs / factor, ys / factor) * factor).astype(np.float32)

    zone = working_zone(width_map, factor, params)
    mask = paint_mask(small, zone, width_map, car_boxes * factor, params)

    # LSD на слегка размытой маске: у бинарной маски резкие края, размытие стабилизирует градиент
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(cv2.GaussianBlur(mask, (3, 3), 0))[0]
    raw = np.zeros((0, 4), dtype=np.float32) if lines is None else lines.reshape(-1, 4)
    segments, paired = merge_edge_pairs(raw, width_at(width_map, raw), params)

    # в сшивку идут только обрывки заметной длины: текстура плитки и листвы даёт россыпь крох,
    # из которых сшивка иначе собирает длинные ложные линии
    pieces = relative_length(segments, scale, factor) >= params.piece_min
    segments, paired = segments[pieces], paired[pieces]
    segments, paired = link_collinear(segments, paired, width_at(width_map, segments), params)

    # фильтр длины разделителя в долях локальной длинной стороны машины
    ratio = relative_length(segments, scale, factor)
    keep = (ratio >= params.seg_len_min) & (ratio <= params.seg_len_max)
    segments, paired = segments[keep] / factor, paired[keep]

    return Markings(
        status="ok" if len(segments) else "failed",
        reason=None if len(segments) else "no_markings",
        scale=scale,
        work_factor=factor,
        zone_mask=zone,
        zone_polygon=zone_outline(zone, factor),
        paint_mask=mask,
        segments=segments,
        paired=paired,
    )


# один разделитель ряда: середина, направление вдоль разметки, источник; у наблюдаемого - концы отрезка
@dataclass(frozen=True)
class Divider:
    mid: np.ndarray
    direction: np.ndarray
    source: Literal["observed", "interpolated", "projected"]
    raw: np.ndarray | None


# принятый ряд: разделители по порядку вдоль ряда, у каждого передний и задний конец
@dataclass(frozen=True)
class Row:
    index: int
    dividers: np.ndarray
    sources: list[str]


# место: четырёхугольник между соседними разделителями ряда; подтверждено, если обе границы
# наблюдаемые либо внутри лежит опорная точка машины
@dataclass(frozen=True)
class Slot:
    row: int
    index: int
    polygon: np.ndarray
    confidence: float
    source: str
    confirmed: bool


# результат калибровки одного или нескольких снимков одной позиции
@dataclass(frozen=True)
class Calibration:
    status: Literal["ok", "partial", "failed"]
    reason: str | None
    markings: Markings
    rows: list[Row]
    slots: list[Slot]
    coverage: float


# кандидат в ряд после заполнения пропусков и хода по сетке: разделители, их концы на линиях ряда
# и места - полигон, уверенность, источник, подтверждение
@dataclass(frozen=True)
class RowCandidate:
    dividers: list[Divider]
    ends: list[np.ndarray]
    slots: list[tuple[np.ndarray, float, str, bool]]
    along: np.ndarray


# направление приводится к одной полуплоскости, чтобы параллельные разделители смотрели в одну сторону
def canonical(direction: np.ndarray) -> np.ndarray:
    flip = (direction[:, 1] < 0) | ((direction[:, 1] == 0) & (direction[:, 0] < 0))
    return np.where(flip[:, None], -direction, direction)


# шаг 6: цепочки соседних разделителей; соседи почти параллельны, отстоят поперёк на [gap_min, gap_max]
# ширины машины и сдвинуты вдоль своего направления не больше чем на row_shift длины машины;
# у разделителя не больше одного ближайшего соседа с каждой стороны, связь только взаимная
def chain_dividers(segments: np.ndarray, scale: ScaleModel, params: CalibrationSettings) -> list[list[int]]:
    count = len(segments)
    if count < 2:
        return []
    p1, p2 = segments[:, :2], segments[:, 2:]
    mid = (p1 + p2) / 2
    length = np.linalg.norm(p2 - p1, axis=1)
    direction = canonical((p2 - p1) / np.maximum(length, 1e-6)[:, None])
    normal = np.stack([-direction[:, 1], direction[:, 0]], 1)
    width = scale.short_at(mid[:, 0], mid[:, 1])
    long_side = scale.long_at(mid[:, 0], mid[:, 1])

    angle_ok = np.abs(direction @ direction.T) >= np.cos(np.radians(params.row_angle_deg))
    offset = mid[None, :, :] - mid[:, None, :]
    across = np.einsum("ijk,ik->ij", offset, normal)
    pair_width = (width[:, None] + width[None, :]) / 2
    gap_ok = (np.abs(across) >= params.gap_min * pair_width) & (np.abs(across) <= params.gap_max * pair_width)
    # сдвиг вдоль общего направления пары симметричен; при косой парковке соседние разделители
    # сдвинуты вдоль себя, поэтому требование перекрытия отрезков не годится
    mean_direction = direction[:, None, :] + direction[None, :, :] * np.sign(direction @ direction.T)[:, :, None]
    mean_direction /= np.maximum(np.linalg.norm(mean_direction, axis=2, keepdims=True), 1e-6)
    shift = np.abs(np.einsum("ijk,ijk->ij", offset, mean_direction))
    shift_ok = shift <= params.row_shift * (long_side[:, None] + long_side[None, :]) / 2
    candidate = angle_ok & gap_ok & shift_ok
    np.fill_diagonal(candidate, False)

    # ближайший сосед с положительной и с отрицательной стороны нормали
    right = np.full(count, -1)
    left = np.full(count, -1)
    for i in range(count):
        for side, target in ((1, right), (-1, left)):
            options = np.flatnonzero(candidate[i] & (side * across[i] > 0))
            if len(options):
                target[i] = options[np.argmin(np.abs(across[i, options]))]
    linked = [bool(right[i] >= 0 and left[right[i]] == i) for i in range(count)]
    has_left = {int(right[i]) for i in range(count) if linked[i]}

    chains = []
    for start in range(count):
        if start in has_left:
            continue
        chain = [start]
        while linked[chain[-1]]:
            chain.append(int(right[chain[-1]]))
        if len(chain) >= 2:
            chains.append(chain)
    return chains


# шаги между соседними разделителями вдоль оси ряда в долях ширины машины в середине пары:
# в этих единицах шаг ряда не зависит от перспективы
def relative_gaps(dividers: list[Divider], across: np.ndarray, scale: ScaleModel) -> np.ndarray:
    mids = np.array([d.mid for d in dividers])
    centers = (mids[:-1] + mids[1:]) / 2
    return np.diff(mids @ across) / scale.short_at(centers[:, 0], centers[:, 1])


# шаг 7: пропуски в целое число медианных шагов заполняются интерполированными разделителями;
# пропуск больше max_fill_gap шагов или не кратный шагу разрывает цепочку на части;
# gaps - шаги между соседними разделителями в долях ширины машины
def fill_gaps(dividers: list[Divider], gaps: np.ndarray, params: CalibrationSettings) -> list[list[Divider]]:
    step = float(np.median(gaps))
    pieces: list[list[Divider]] = [[dividers[0]]]
    for a, b, gap in zip(dividers, dividers[1:], gaps):
        ratio = gap / step
        k = int(round(ratio))
        if k >= 2 and (k > params.max_fill_gap or abs(ratio - k) > FILL_TOLERANCE):
            pieces.append([b])
            continue
        for s in range(1, k):
            t = s / k
            direction = a.direction * (1 - t) + b.direction * t
            direction = direction / np.linalg.norm(direction)
            pieces[-1].append(Divider(a.mid * (1 - t) + b.mid * t, direction, "interpolated", None))
        pieces[-1].append(b)
    return pieces


# передняя или задняя линия ряда: полином по положению вдоль ряда с одним проходом отсева выбросов,
# чтобы концы частично закрытых машинами разделителей не тянули линию на себя; концы, обрезанные
# краем кадра, в подгонку не идут, а если настоящих концов меньше двух - линия идёт по крайнему из всех
def fit_edge(u: np.ndarray, v: np.ndarray, cut: np.ndarray, tolerance: float, outer: int) -> np.poly1d:
    if (~cut).sum() < 2:
        return np.poly1d([float(v.max() if outer > 0 else v.min())])
    u, v = u[~cut], v[~cut]
    degree = 2 if len(u) >= 5 else 1
    poly = np.poly1d(np.polyfit(u, v, degree))
    keep = np.abs(poly(u) - v) <= tolerance
    if keep.sum() >= degree + 1 and not keep.all():
        poly = np.poly1d(np.polyfit(u[keep], v[keep], degree))
    return poly


# линия ряда как функция положения вдоль ряда: внутри наблюдаемого участка [low, high] - полином,
# за его пределами - касательная в крайней точке, потому что парабола вне своих точек быстро уходит
def tangent_beyond(poly: np.poly1d, low: float, high: float) -> Callable[[float], float]:
    slope = poly.deriv()

    def line(u: float) -> float:
        edge = min(max(u, low), high)
        return float(poly(edge) + slope(edge) * (u - edge))

    return line


# концы разделителя на передней и задней линиях ряда
def divider_ends(
    divider: Divider,
    along: np.ndarray,
    across: np.ndarray,
    front: Callable[[float], float],
    back: Callable[[float], float],
) -> np.ndarray:
    direction = divider.direction if divider.direction @ along >= 0 else -divider.direction
    u = float(divider.mid @ across)
    v = float(divider.mid @ along)
    speed = max(float(direction @ along), 1e-6)
    return np.concatenate([
        divider.mid + (front(u) - v) / speed * direction,
        divider.mid + (back(u) - v) / speed * direction,
    ])


# шаг 8: место между соседними разделителями, если четырёхугольник выпуклый и разумной площади
def slot_polygon(a: np.ndarray, b: np.ndarray, scale: ScaleModel, params: CalibrationSettings) -> np.ndarray | None:
    polygon = np.array([a[:2], a[2:], b[2:], b[:2]], dtype=np.float32)
    if not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
        return None
    center = polygon.mean(0)
    width = float(scale.short_at(center[0], center[1]))
    area = cv2.contourArea(polygon) / width**2
    return polygon if params.slot_area_min <= area <= params.slot_area_max else None


# шаг 9: уверенность и источник места по качеству его двух границ; машина в ячейке отличает
# только подтверждённую проекцию от неподтверждённой
def slot_quality(
    a: Divider, b: Divider, ends_a: np.ndarray, ends_b: np.ndarray, occupied: bool
) -> tuple[float, str]:
    sources = {a.source, b.source}
    if "projected" in sources:
        return (CONFIDENCE_PROJECTED, "projected") if occupied else (CONFIDENCE_WEAK, "extrapolated")
    if sources == {"interpolated"}:
        return CONFIDENCE_WEAK, "interpolated"
    if "interpolated" in sources:
        return CONFIDENCE_ONE_INTERPOLATED, "interpolated"
    # у двух наблюдаемых границ уверенность - видимая доля их длины на месте
    visible = []
    for divider, ends in ((a, ends_a), (b, ends_b)):
        seen = float(np.linalg.norm(divider.raw[2:] - divider.raw[:2]))
        visible.append(min(1.0, seen / max(float(np.linalg.norm(ends[2:] - ends[:2])), 1e-6)))
    return float(np.mean(visible)), "observed"


# лежит ли точка исходного кадра в рабочей зоне
def in_zone(markings: Markings, point: np.ndarray) -> bool:
    x, y = (point * markings.work_factor).astype(int)
    height, width = markings.zone_mask.shape
    return bool(0 <= x < width and 0 <= y < height and markings.zone_mask[y, x])


# шаг 7, ход по сетке за крайний разделитель ряда: side -1 - к началу ряда, +1 - к концу;
# следующая граница ставится через шаг ряда; ячейка принимается по опорной точке машины внутри
# либо по наблюдаемому разделителю вне цепочки на её дальней границе; free - такие разделители,
# уже отобранные по углу; ends_of - концы разделителя на линиях ряда
def extend_row(
    piece: list[Divider],
    side: int,
    step_w: float,
    along: np.ndarray,
    across: np.ndarray,
    ends_of: Callable[[Divider], np.ndarray],
    free: list[Divider],
    cars: np.ndarray,
    markings: Markings,
    params: CalibrationSettings,
) -> list[Divider]:
    scale = markings.scale
    prev = piece[0] if side < 0 else piece[-1]
    walked: list[Divider] = []
    pending: list[Divider] = []
    while len(pending) < params.max_projected_run:
        # шаг в пикселях - по ширине машины в центре будущей ячейки
        probe = prev.mid + side * 0.5 * step_w * float(scale.short_at(prev.mid[0], prev.mid[1])) * across
        step = step_w * float(scale.short_at(probe[0], probe[1]))
        border = Divider(prev.mid + side * step * across, along, "projected", None)

        # наблюдаемый разделитель на месте границы: граница встаёт на него; сдвиг вдоль разметки
        # ограничен, иначе ход цепляется за разделитель соседнего ряда выше или ниже
        center = ends_of(border).reshape(2, 2).mean(0)
        shift_limit = params.row_shift * float(scale.long_at(center[0], center[1]))
        near = [
            d
            for d in free
            if abs(float((d.mid - center) @ across)) <= FILL_TOLERANCE * step
            and abs(float((d.mid - center) @ along)) <= shift_limit
        ]
        snapped = min(near, key=lambda d: abs(float((d.mid - center) @ across)), default=None)
        if snapped is not None:
            border = snapped

        polygon = slot_polygon(ends_of(prev), ends_of(border), scale, params)
        if polygon is None or not in_zone(markings, polygon.mean(0)):
            break
        inside = [car for car in cars if contains(polygon, car)]
        if inside and snapped is None:
            # подтяжка: проекция опорной точки ближайшей к центру машины ложится посередине ячейки,
            # это гасит накопление ошибки на фишае и на искривлении ряда
            middle = float(polygon.mean(0) @ across)
            car = min(inside, key=lambda c: abs(float(c @ across) - middle))
            width = 2 * abs(float((car - prev.mid) @ across))
            width = float(np.clip(width, PULL_RANGE[0] * step, PULL_RANGE[1] * step))
            border = Divider(prev.mid + side * width * across, along, "projected", None)

        # не принятая ячейка остаётся местом, только если за ней нашлась принятая
        pending.append(border)
        if inside or snapped is not None:
            walked.extend(pending)
            pending = []
        prev = border
    return [*reversed(walked), *piece] if side < 0 else [*piece, *walked]


# шаги 6-9 для одной цепочки: шаг ряда, заполнение пропусков, линии ряда, ход по сетке, места;
# frame - ширина и высота исходного кадра
def build_rows(
    chain: list[int],
    markings: Markings,
    cars: np.ndarray,
    frame: tuple[int, int],
    params: CalibrationSettings,
) -> list[RowCandidate]:
    segments, scale = markings.segments, markings.scale
    raw = segments[chain]
    length = np.linalg.norm(raw[:, 2:] - raw[:, :2], axis=1)
    directions = canonical((raw[:, 2:] - raw[:, :2]) / length[:, None])
    along = directions.mean(0)
    along /= np.linalg.norm(along)
    # ось вдоль ряда смотрит вправо, у вертикального ряда - вниз: места нумеруются слева направо
    across = np.array([-along[1], along[0]])
    if across[0] < 0 or (across[0] == 0 and across[1] < 0):
        across = -across
    dividers = [Divider((r[:2] + r[2:]) / 2, d, "observed", r) for r, d in zip(raw, directions)]
    dividers.sort(key=lambda d: float(d.mid @ across))

    # наблюдаемые разделители вне цепочки, почти параллельные ряду: к ним привязывается ход по сетке
    rest = np.delete(segments, chain, axis=0)
    rest_length = np.linalg.norm(rest[:, 2:] - rest[:, :2], axis=1)
    rest_directions = canonical((rest[:, 2:] - rest[:, :2]) / np.maximum(rest_length, 1e-6)[:, None])
    parallel = np.abs(rest_directions @ along) >= np.cos(np.radians(params.row_angle_deg))
    free = [
        Divider((r[:2] + r[2:]) / 2, d, "observed", r) for r, d in zip(rest[parallel], rest_directions[parallel])
    ]

    rows = []
    for piece in fill_gaps(dividers, relative_gaps(dividers, across, scale), params):
        observed = [d for d in piece if d.source == "observed"]
        if len(observed) < params.row_min_dividers:
            continue
        # шаг ряда в долях ширины машины; у пары разделителей вариация шага не определена,
        # поэтому её шаг зажат в узкий диапазон одного места
        gaps = relative_gaps(piece, across, scale)
        step_w = float(np.median(gaps))
        if np.std(gaps) / np.mean(gaps) > params.row_gap_cv:
            continue
        if len(observed) == 2 and not params.pair_gap_min <= step_w <= params.pair_gap_max:
            continue

        # передняя и задняя линии по концам наблюдаемых разделителей; конец у края кадра - обрезанный
        u = np.array([d.mid @ across for d in observed])
        points = np.array([[d.raw[:2], d.raw[2:]] for d in observed])
        values = points @ along
        order = np.argsort(values, axis=1)
        ends_v = np.take_along_axis(values, order, axis=1)
        margin = FRAME_MARGIN * max(frame)
        at_border = (
            (points.min(axis=2) <= margin)
            | (points[:, :, 0] >= frame[0] - margin)
            | (points[:, :, 1] >= frame[1] - margin)
        )
        cut = np.take_along_axis(at_border, order, axis=1)
        mids = np.array([d.mid for d in observed])
        tolerance = 0.15 * float(np.median(scale.long_at(mids[:, 0], mids[:, 1])))
        low, high = float(u.min()), float(u.max())
        front = tangent_beyond(fit_edge(u, ends_v[:, 0], cut[:, 0], tolerance, -1), low, high)
        back = tangent_beyond(fit_edge(u, ends_v[:, 1], cut[:, 1], tolerance, 1), low, high)

        # концы разделителя на линиях этого ряда
        def ends_of(divider: Divider) -> np.ndarray:
            return divider_ends(divider, along, across, front, back)

        for side in (-1, 1):
            piece = extend_row(piece, side, step_w, along, across, ends_of, free, cars, markings, params)

        ends = [ends_of(d) for d in piece]
        slots = []
        for a, b, ends_a, ends_b in zip(piece, piece[1:], ends, ends[1:]):
            polygon = slot_polygon(ends_a, ends_b, scale, params)
            if polygon is not None:
                occupied = any(contains(polygon, car) for car in cars)
                confidence, source = slot_quality(a, b, ends_a, ends_b, occupied)
                slots.append((polygon, confidence, source, source == "observed" or occupied))
        if slots:
            rows.append(RowCandidate(piece, ends, slots, along))
    return rows


# приёмка рядов: между разделителями стоит машина, либо ряд почти параллелен уже принятому
# и отстоит от него не дальше row_neighbour_gap длин машины; принятие распространяется по соседям
def accept_rows(
    candidates: list[RowCandidate], cars: np.ndarray, scale: ScaleModel, params: CalibrationSettings
) -> list[RowCandidate]:
    accepted = [any(contains(slot[0], car) for slot in row.slots for car in cars) for row in candidates]
    mids = [np.array([d.mid for d in row.dividers]) for row in candidates]
    cos_limit = np.cos(np.radians(params.row_angle_deg))
    changed = True
    while changed:
        changed = False
        for i, row in enumerate(candidates):
            if accepted[i]:
                continue
            for j, other in enumerate(candidates):
                if not accepted[j] or abs(float(row.along @ other.along)) < cos_limit:
                    continue
                distances = np.linalg.norm(mids[i][:, None, :] - mids[j][None, :, :], axis=2)
                a, b = np.unravel_index(np.argmin(distances), distances.shape)
                point = (mids[i][a] + mids[j][b]) / 2
                if distances[a, b] <= params.row_neighbour_gap * float(scale.long_at(point[0], point[1])):
                    accepted[i] = changed = True
                    break
    return [row for row, ok in zip(candidates, accepted) if ok]


# статус калибровки по 7.2 «Деградация» и покрытие - доля подтверждённых мест
def calibration_status(
    slots: list[Slot], params: CalibrationSettings
) -> tuple[Literal["ok", "partial", "failed"], str | None, float]:
    if not slots:
        return "failed", "no_rows", 0.0
    coverage = sum(s.confirmed for s in slots) / len(slots)
    if len(slots) < params.min_slots:
        return "partial", "few_slots", coverage
    if coverage < COVERAGE_OK:
        return "partial", "low_coverage", coverage
    return "ok", None, coverage


# шаги 6-9: из найденной разметки и машин - ряды и места
def build_slots(
    markings: Markings, car_boxes: np.ndarray, params: CalibrationSettings, ground_offset: float
) -> Calibration:
    if markings.status == "failed":
        return Calibration("failed", markings.reason, markings, [], [], 0.0)
    cars = ground_points(car_boxes, ground_offset)
    height, width = markings.zone_mask.shape
    frame = (round(width / markings.work_factor), round(height / markings.work_factor))
    candidates = []
    for chain in chain_dividers(markings.segments, markings.scale, params):
        candidates.extend(build_rows(chain, markings, cars, frame, params))
    accepted = accept_rows(candidates, cars, markings.scale, params)

    # встречный ход двух цепочек одного ряда даёт дубли: место, центр которого лежит в месте ряда
    # с большим числом наблюдаемых разделителей, убирается
    accepted.sort(key=lambda r: -sum(d.source == "observed" for d in r.dividers))
    taken: list[np.ndarray] = []
    unique = []
    for row in accepted:
        kept = [s for s in row.slots if not any(contains(polygon, s[0].mean(0)) for polygon in taken)]
        taken.extend(s[0] for s in kept)
        if kept:
            unique.append(RowCandidate(row.dividers, row.ends, kept, row.along))

    # ряды нумеруются сверху вниз по средней высоте, места в ряду - по порядку вдоль ряда
    unique.sort(key=lambda r: float(np.mean([d.mid[1] for d in r.dividers])))
    rows, slots = [], []
    for index, row in enumerate(unique):
        rows.append(Row(index, np.array(row.ends, dtype=np.float32), [d.source for d in row.dividers]))
        for number, (polygon, confidence, source, confirmed) in enumerate(row.slots):
            slots.append(Slot(index, number, polygon, confidence, source, confirmed))
    status, reason, coverage = calibration_status(slots, params)
    return Calibration(status, reason, markings, rows, slots, coverage)


# полная калибровка одного RGB-снимка по найденным боксам машин
def calibrate(
    image: np.ndarray, car_boxes: np.ndarray, params: CalibrationSettings, ground_offset: float
) -> Calibration:
    return build_slots(find_markings(image, car_boxes, params), car_boxes, params, ground_offset)


# несколько снимков с одной позиции: места объединяются по IoU, confidence - доля снимков с этим местом
def merge_calibrations(calibrations: list[Calibration], params: CalibrationSettings) -> Calibration:
    usable = [c for c in calibrations if c.slots]
    if len(calibrations) == 1 or not usable:
        return calibrations[0]
    reference = max(usable, key=lambda c: len(c.slots))
    merged = [[slot, 1] for slot in reference.slots]
    rows = list(reference.rows)
    for calibration in usable:
        if calibration is reference:
            continue
        # места, не совпавшие с уже известными, добавляются со своими рядами под новыми номерами
        offset = len(rows)
        for slot in calibration.slots:
            ious = [polygon_iou(known.polygon, slot.polygon) for known, _ in merged]
            best = int(np.argmax(ious))
            if ious[best] >= MERGE_IOU:
                merged[best][1] += 1
            else:
                moved = Slot(slot.row + offset, slot.index, slot.polygon, slot.confidence, slot.source, slot.confirmed)
                merged.append([moved, 1])
        rows.extend(Row(r.index + offset, r.dividers, r.sources) for r in calibration.rows)
    slots = [
        Slot(s.row, s.index, s.polygon, count / len(calibrations), s.source, s.confirmed) for s, count in merged
    ]
    status, reason, coverage = calibration_status(slots, params)
    return Calibration(status, reason, reference.markings, rows, slots, coverage)
