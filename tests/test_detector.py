from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from parkingscan.config import InferenceConfig, PostprocessConfig
from parkingscan.core import detector as detector_module
from parkingscan.core.detector import MODEL_CONF, CarDetector, to_model_scale

# контракт как в models/config.json, но собранный в тесте, чтобы не зависеть от файлов
CONFIG = InferenceConfig(
    model_file="cars_yolo26s_best.pt",
    scale=0.5,
    tile=640,
    overlap=0.25,
    postprocess=PostprocessConfig(
        postprocess_type="NMS",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.6,
    ),
    vis_conf=0.3,
)

# кадр того же размера, что и в обучающем датасете
FRAME = np.zeros((3000, 4000, 3), dtype=np.uint8)


# предсказание SAHI в минимальном виде, который читает детектор
def fake_prediction(box: list[float], score: float) -> SimpleNamespace:
    return SimpleNamespace(bbox=SimpleNamespace(to_xyxy=lambda: box), score=SimpleNamespace(value=score))


# детектор с подменёнными моделью и SAHI: без GPU, весов и сети
@pytest.fixture
def fake_detector(monkeypatch: pytest.MonkeyPatch) -> tuple[CarDetector, dict]:
    calls: dict = {}
    predictions = [
        fake_prediction([100.0, 50.0, 300.0, 150.0], 0.9),
        fake_prediction([10.0, 20.0, 30.0, 40.0], 0.2),
    ]

    def fake_from_pretrained(**kwargs: object) -> object:
        calls["model_kwargs"] = kwargs
        return object()

    def fake_sliced_prediction(image: np.ndarray, model: object, **kwargs: object) -> SimpleNamespace:
        calls["image_shape"] = image.shape
        calls["sahi_kwargs"] = kwargs
        return SimpleNamespace(object_prediction_list=predictions)

    monkeypatch.setattr(detector_module.AutoDetectionModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(detector_module, "get_sliced_prediction", fake_sliced_prediction)
    return CarDetector(Path("model.pt"), CONFIG, "cpu"), calls


# кадр 4000x3000 уменьшается до 2000 по большей стороне с сохранением пропорций
def test_to_model_scale_downscales_to_target_side() -> None:
    resized, k = to_model_scale(FRAME, 2000)
    assert k == 0.5
    assert resized.shape == (1500, 2000, 3)


# маленький кадр увеличивается, портретная ориентация масштабируется по высоте
def test_to_model_scale_upscales_portrait() -> None:
    image = np.zeros((1000, 500, 3), dtype=np.uint8)
    resized, k = to_model_scale(image, 2000)
    assert k == 2.0
    assert resized.shape == (2000, 1000, 3)


# боксы возвращаются в координатах исходного кадра, float32, отфильтрованные по переданному порогу
def test_detect_maps_boxes_back_and_filters_by_conf(fake_detector: tuple[CarDetector, dict]) -> None:
    detector, calls = fake_detector
    boxes, scores = detector.detect(FRAME, 0.3)
    assert calls["image_shape"] == (1500, 2000, 3)
    assert boxes.dtype == np.float32
    assert scores.dtype == np.float32
    np.testing.assert_allclose(boxes, [[200.0, 100.0, 600.0, 300.0]])
    np.testing.assert_allclose(scores, [0.9])


# пониженный порог пропускает слабые детекции
def test_detect_lower_conf_keeps_weak_boxes(fake_detector: tuple[CarDetector, dict]) -> None:
    detector, _ = fake_detector
    boxes, _ = detector.detect(FRAME, 0.15)
    assert len(boxes) == 2
    np.testing.assert_allclose(boxes[1], [20.0, 40.0, 60.0, 80.0])


# пустой результат SAHI даёт массивы правильной формы, а не ошибку
def test_detect_empty_result(fake_detector: tuple[CarDetector, dict], monkeypatch: pytest.MonkeyPatch) -> None:
    detector, _ = fake_detector
    monkeypatch.setattr(
        detector_module,
        "get_sliced_prediction",
        lambda image, model, **kwargs: SimpleNamespace(object_prediction_list=[]),
    )
    boxes, scores = detector.detect(FRAME, 0.3)
    assert boxes.shape == (0, 4)
    assert scores.shape == (0,)


# параметры модели и SAHI берутся из контракта, порог модели - MODEL_CONF
def test_detect_uses_contract_parameters(fake_detector: tuple[CarDetector, dict]) -> None:
    detector, calls = fake_detector
    detector.detect(FRAME, 0.3)
    assert calls["model_kwargs"]["confidence_threshold"] == MODEL_CONF
    assert calls["model_kwargs"]["image_size"] == 640
    sahi_kwargs = calls["sahi_kwargs"]
    assert sahi_kwargs["slice_height"] == sahi_kwargs["slice_width"] == 640
    assert sahi_kwargs["overlap_height_ratio"] == sahi_kwargs["overlap_width_ratio"] == 0.25
    assert sahi_kwargs["perform_standard_pred"] is True
    assert sahi_kwargs["postprocess_type"] == "NMS"
    assert sahi_kwargs["postprocess_match_metric"] == "IOS"
    assert sahi_kwargs["postprocess_match_threshold"] == 0.6
