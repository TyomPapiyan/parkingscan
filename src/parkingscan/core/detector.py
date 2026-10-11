import os
import threading
from pathlib import Path

# ultralytics при импорте пишет свои настройки в каталог пользователя (на Windows - в APPDATA);
# каталог переносится внутрь проекта и создаётся заранее: в несуществующий ultralytics не пишет.
# Переменная задаётся до импорта sahi, который подтягивает ultralytics
Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("data") / "ultralytics"))).mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

from parkingscan.config import InferenceConfig

# Порог уверенности внутри модели. Метрики из models/config.json (mAP50, best_F1) и сам vis_conf
# измерены в ноутбуке именно так: модель отдаёт кандидатов от 0.01, SAHI склеивает тайлы,
# и только после этого результат отсекается по рабочему порогу. Не поднимать до vis_conf:
# для другой схемы инференса заявленные метрики не измерялись
MODEL_CONF = 0.01

# сколько тайлов уходит в модель за один вызов, влияет только на скорость
SLICE_BATCH = 4


# первая видеокарта CUDA, если доступна, иначе CPU; возвращает идентификатор и имя устройства
def select_device() -> tuple[str, str]:
    if torch.cuda.is_available():
        return "cuda:0", torch.cuda.get_device_name(0)
    return "cpu", "CPU"


# приводим изображение к масштабу, на котором обучалась модель
def to_model_scale(image: np.ndarray, target_side: int) -> tuple[np.ndarray, float]:
    # коэффициент приведения по большей стороне
    k = target_side / max(image.shape[:2])
    # уменьшение делаем area-интерполяцией, увеличение - кубической
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(image, None, fx=k, fy=k, interpolation=interp)
    return resized, k


# детектор машин на аэрофото: YOLO26s + тайловый инференс SAHI по контракту models/config.json
class CarDetector:
    def __init__(self, model_path: Path, config: InferenceConfig, device: str) -> None:
        self.config = config
        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(model_path),
            confidence_threshold=MODEL_CONF,
            device=device,
            image_size=config.tile,
        )
        # запросы API приходят из пула потоков, а модель на одном GPU не потокобезопасна;
        # одна блокировка сериализует инференс, при росте нагрузки - очередь с отдельным воркером
        self._lock = threading.Lock()

    # холостой прогон на пустом кадре рабочего размера: первый инференс на CUDA
    # инициализирует ядра и длится больше секунды, пусть это случится до первого запроса
    def warmup(self) -> None:
        side = self.config.target_side
        self.detect(np.zeros((side * 3 // 4, side, 3), dtype=np.uint8), MODEL_CONF)

    # детекция на RGB-изображении; боксы xyxy в координатах исходного изображения и их conf;
    # рабочий порог conf обязателен - его задаёт приложение, своего значения по умолчанию у детектора нет
    def detect(self, image: np.ndarray, conf: float) -> tuple[np.ndarray, np.ndarray]:
        small, k = to_model_scale(image, self.config.target_side)
        tile = self.config.tile
        overlap = self.config.overlap
        postprocess = self.config.postprocess
        with self._lock:
            result = get_sliced_prediction(
                small,
                self.sahi_model,
                slice_height=tile,
                slice_width=tile,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                perform_standard_pred=True,
                postprocess_type=postprocess.postprocess_type,
                postprocess_match_metric=postprocess.postprocess_match_metric,
                postprocess_match_threshold=postprocess.postprocess_match_threshold,
                batch_size=SLICE_BATCH,
                verbose=0,
            )
        predictions = result.object_prediction_list
        # координаты SAHI в системе уменьшенного кадра, делением на k возвращаемся к исходному
        boxes = np.array([p.bbox.to_xyxy() for p in predictions], dtype=np.float32).reshape(-1, 4) / k
        scores = np.array([p.score.value for p in predictions], dtype=np.float32)
        keep = scores >= conf
        return boxes[keep], scores[keep]
