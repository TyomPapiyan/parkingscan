import argparse
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from parkingscan.config import load_inference_config
from parkingscan.core.detector import MODEL_CONF, CarDetector, select_device

# Ручная проверка детектора против разметки.
# Без аргумента - 4 эталонных снимка из data/samples:
#   uv run python tests/check_detector.py
# С путём к папке датасета (внутри images/ и labels/) - все снимки val-сплита ноутбука:
#   uv run python tests/check_detector.py ../43-CarsObjectDetection/train/train
# Имя без префикса test_, поэтому pytest этот скрипт не собирает: ему нужны веса и GPU

# корень репозитория, скрипт лежит в tests/
ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "data" / "samples"
MODELS_DIR = ROOT / "models"

# val-сплит как в ячейке 5 ноутбука: сортировка, random.seed(42), shuffle, первые 15%
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_SEED = 42
VAL_FRAC = 0.15

# бокс считается найденным при IoU с эталоном не ниже этого значения
IOU_MATCH = 0.5

# критерий приёмки Этапа 1 при пороге режима подсчёта: медиана абсолютного отклонения
# счёта по снимкам и F1 боксов по всем снимкам; поснимочный допуск не годится - на кадре
# с двумя машинами один промах даёт 50%, и метрика мерит разброс датасета, а не детектор
COUNT_CONF = 0.25
MEDIAN_DEV_MAX = 0.10
F1_MIN = 0.83

# пороги прогона; третий подставляется точным vis_conf из контракта
SWEEP = (0.15, COUNT_CONF, None, 0.40)

# при большем числе снимков построчные данные не печатаются, только итоги
PER_IMAGE_LIMIT = 6

# ориентиры из ноутбука, посчитанные по всем 35 снимкам val-сплита
NOTEBOOK_F1 = 0.844
NOTEBOOK_RECALL = 0.75


# попарный IoU между двумя наборами боксов xyxy
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


# число верных детекций: жадно по убыванию conf, каждый эталонный бокс используется один раз
def true_positives(boxes: np.ndarray, scores: np.ndarray, gt: np.ndarray) -> int:
    if len(boxes) == 0 or len(gt) == 0:
        return 0
    ious = iou_matrix(boxes[np.argsort(-scores)], gt)
    used = np.zeros(len(gt), dtype=bool)
    tp = 0
    for row in ious:
        candidates = np.where(used, -1.0, row)
        j = int(candidates.argmax())
        if candidates[j] >= IOU_MATCH:
            used[j] = True
            tp += 1
    return tp


# итог по снимку или по сумме снимков: число найденных, эталон, верные, время
class Score:
    def __init__(self, found: int, expected: int, tp: int, ms: float) -> None:
        self.found = found
        self.expected = expected
        self.tp = tp
        self.ms = ms

    # относительное отклонение счёта машин от эталона
    @property
    def deviation(self) -> float:
        return (self.found - self.expected) / self.expected

    @property
    def precision(self) -> float:
        return self.tp / self.found if self.found else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.expected

    @property
    def f1(self) -> float:
        return 2 * self.tp / (self.found + self.expected)


# сумма по нескольким снимкам: счёт и TP складываются, время усредняется
def total(scores: list[Score]) -> Score:
    return Score(
        found=sum(s.found for s in scores),
        expected=sum(s.expected for s in scores),
        tp=sum(s.tp for s in scores),
        ms=statistics.mean(s.ms for s in scores),
    )


# прогон детектора с замером времени и сравнением с эталоном
def evaluate(detector: CarDetector, image: np.ndarray, gt: np.ndarray, conf: float) -> Score:
    start = time.perf_counter()
    boxes, scores = detector.detect(image, conf)
    ms = (time.perf_counter() - start) * 1000
    return Score(len(boxes), len(gt), true_positives(boxes, scores, gt), ms)


# markdown-таблица в stdout
def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


# ячейки таблицы с метриками одного варианта
def score_cells(s: Score) -> list[str]:
    return [str(s.found), f"{s.deviation:+.1%}", f"{s.precision:.4f}", f"{s.recall:.4f}", f"{s.f1:.4f}", f"{s.ms:.0f}"]


# RGB-изображение с диска
def load_rgb(path: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# эталонные снимки из data/samples с боксами из ground_truth.json
def load_samples() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    ground_truth = json.loads((SAMPLES_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    return {
        name: (load_rgb(SAMPLES_DIR / name), np.array(item["boxes"], dtype=np.float32).reshape(-1, 4))
        for name, item in ground_truth.items()
    }


# разметка YOLO (класс, cx, cy, w, h в долях кадра) в боксы xyxy в пикселях, как labels_to_xyxy
def read_yolo_boxes(label_path: Path, width: int, height: int) -> np.ndarray:
    rows = [line.split()[1:5] for line in label_path.read_text().splitlines() if line.strip()]
    lab = np.array(rows, dtype=np.float32).reshape(-1, 4)
    cx, cy, bw, bh = lab[:, 0] * width, lab[:, 1] * height, lab[:, 2] * width, lab[:, 3] * height
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


# снимки val-сплита ноутбука с боксами из labels/<stem>.txt
def load_val_split(dataset: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    images = sorted(p for p in dataset.rglob("*") if p.suffix.lower() in IMG_EXT)
    labels_dir = next(p for p in dataset.rglob("labels") if p.is_dir())
    random.seed(SPLIT_SEED)
    random.shuffle(images)
    val = images[: round(len(images) * VAL_FRAC)]
    samples = {}
    for path in val:
        image = load_rgb(path)
        height, width = image.shape[:2]
        samples[path.name] = (image, read_yolo_boxes(labels_dir / f"{path.stem}.txt", width, height))
    return samples


# детерминизм, сравнение двух вариантов порога модели и подбор порога для подсчёта
def main() -> None:
    parser = argparse.ArgumentParser(description="Check the car detector against ground truth")
    parser.add_argument("dataset", nargs="?", type=Path, help="dataset folder with images/ and labels/")
    args = parser.parse_args()

    config = load_inference_config(MODELS_DIR / "config.json")
    device_id, device_name = select_device()
    detector = CarDetector(MODELS_DIR / config.model_file, config, device_id)
    samples = load_val_split(args.dataset) if args.dataset else load_samples()
    names = list(samples)
    images = {n: samples[n][0] for n in names}
    gts = {n: samples[n][1] for n in names}
    n_images = len(names)
    per_image = n_images <= PER_IMAGE_LIMIT
    source = f"val split of {args.dataset}" if args.dataset else "data/samples"
    print(f"Device: {device_id} ({device_name}), vis_conf = {config.vis_conf:.4f}")
    print(f"Images: {n_images} from {source}, cars in ground truth: {sum(len(g) for g in gts.values())}\n")

    # первый инференс инициализирует CUDA и в замер не входит
    detector.warmup()

    # проверка детерминизма: самый плотный снимок дважды, сырой выход от порога модели
    densest = max(names, key=lambda n: len(gts[n]))
    first = detector.detect(images[densest], MODEL_CONF)
    second = detector.detect(images[densest], MODEL_CONF)
    same = all(a.tobytes() == b.tobytes() for a, b in zip(first, second))
    print(f"Determinism on {densest}: {len(first[0])} raw boxes, byte-identical: {same}\n")

    # вариант 1 - рабочий: порог модели MODEL_CONF, фильтр по порогу после SAHI
    thresholds = [config.vis_conf if t is None else t for t in SWEEP]
    sweep = {t: {n: evaluate(detector, images[n], gts[n], t) for n in names} for t in thresholds}
    variant1 = sweep[config.vis_conf]

    # вариант 2 - только для сравнения: порог модели равен vis_conf
    detector.sahi_model.confidence_threshold = config.vis_conf
    variant2 = {n: evaluate(detector, images[n], gts[n], config.vis_conf) for n in names}
    detector.sahi_model.confidence_threshold = MODEL_CONF

    print(f"## Variant 1 (model conf {MODEL_CONF}) vs variant 2 (model conf = vis_conf), filter at vis_conf\n")
    metric_headers = ["found", "dev", "P", "R", "F1", "ms"]
    rows = []
    if per_image:
        rows = [[n, str(len(gts[n]))] + score_cells(variant1[n]) + score_cells(variant2[n]) for n in names]
    totals = score_cells(total(list(variant1.values()))) + score_cells(total(list(variant2.values())))
    rows.append(["TOTAL", str(sum(len(g) for g in gts.values()))] + totals)
    print_table(["image", "GT"] + [f"v1 {h}" for h in metric_headers] + [f"v2 {h}" for h in metric_headers], rows)

    print(f"## Threshold sweep, variant 1: count deviation per image and box metrics over {n_images} images\n")
    rows = []
    medians = {}
    f1s = {}
    for t, results in sweep.items():
        devs = [results[n].deviation for n in names]
        agg = total(list(results.values()))
        medians[t] = statistics.median(abs(d) for d in devs)
        f1s[t] = agg.f1
        passed = medians[t] <= MEDIAN_DEV_MAX and f1s[t] >= F1_MIN
        cells = [f"{d:+.1%}" for d in devs] if per_image else []
        rows.append(
            [f"{t:.3f}"]
            + cells
            + [f"{max(abs(d) for d in devs):.1%}", f"{statistics.mean(abs(d) for d in devs):.1%}"]
            + [f"{medians[t]:.1%}", f"{agg.deviation:+.1%}"]
            + [f"{agg.precision:.4f}", f"{agg.recall:.4f}", f"{agg.f1:.4f}", "PASS" if passed else "FAIL"]
        )
    short = [f"{n.split('-')[1].split('.')[0]} ({len(gts[n])})" for n in names] if per_image else []
    headers = ["max abs dev", "mean abs dev", "median abs dev", "total dev", "P", "R", "F1", "verdict"]
    print_table(["conf"] + short + headers, rows)

    # лучший порог для подсчёта: наименьшая медиана отклонения среди порогов, где F1 не ниже нормы
    eligible = [t for t in thresholds if f1s[t] >= F1_MIN] or thresholds
    best = min(eligible, key=lambda t: medians[t])
    agg1 = total(list(variant1.values()))
    print(f"Best threshold for counting: {best:.3f}")
    print(
        f"Notebook reference on 35 val images: F1 {NOTEBOOK_F1}, recall {NOTEBOOK_RECALL}; "
        f"here on {n_images} images: F1 {agg1.f1:.4f}, recall {agg1.recall:.4f}"
    )

    # вердикт Этапа 1 при пороге режима подсчёта
    passed = medians[COUNT_CONF] <= MEDIAN_DEV_MAX and f1s[COUNT_CONF] >= F1_MIN
    print(
        f"Stage 1 at conf {COUNT_CONF}: median abs dev {medians[COUNT_CONF]:.1%} (<= {MEDIAN_DEV_MAX:.0%}), "
        f"F1 {f1s[COUNT_CONF]:.4f} (>= {F1_MIN}) -> {'PASS' if passed else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
