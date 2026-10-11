import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# YAML с настройками по умолчанию, путь можно переопределить переменной окружения
DEFAULT_CONFIG_FILE = Path("configs/default.yaml")
CONFIG_FILE_ENV = "PARKINGSCAN_CONFIG_FILE"

# большая сторона кадров обучающего датасета, scale в config.json задан относительно неё
TRAIN_FRAME_SIDE = 4000


# параметры постобработки SAHI, передаются в get_sliced_prediction как есть
class PostprocessConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    postprocess_type: str
    postprocess_match_metric: str
    postprocess_match_threshold: float


# контракт инференса из models/config.json, сохранённый при обучении модели
class InferenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_file: str
    scale: float
    tile: PositiveInt
    overlap: float
    postprocess: PostprocessConfig
    vis_conf: float

    # большая сторона изображения, к которой приводится вход перед инференсом
    @property
    def target_side(self) -> int:
        return round(TRAIN_FRAME_SIDE * self.scale)


# параметры поиска разметки, шаги 1-5 раздела 7.2; доли - от локального размера машины
class CalibrationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    # большая сторона кадра, на котором ищется разметка, в пикселях
    work_side: PositiveInt = 2000
    # минимум машин для модели масштаба
    scale_min_cars: PositiveInt = 5
    # рабочая зона: ожидаемая ширина машины в пикселях исходного кадра и доля полудиагонали
    roi_min_car_px: float = Field(default=150.0, gt=0)
    roi_max_radius: float = Field(default=0.9, gt=0, le=1.5)
    # маска краски: ядро top-hat, локальный порог в стандартных отклонениях top-hat над средним
    # окрестности, минимальный контраст по каналу L
    tophat_kernel: float = Field(default=0.25, gt=0)
    paint_sigma: float = Field(default=2.0, ge=0)
    paint_min_contrast: float = Field(default=20.0, ge=0)
    # белая краска - низкая насыщенность по HSV (0-255); жёлтая пропускается отдельно по тону
    paint_max_saturation: int = Field(default=60, ge=0, le=255)
    # расширение боксов машин при вычитании и максимальная ширина штриха
    car_mask_dilate: float = Field(default=0.1, ge=0)
    # доля высоты бокса машины снизу, которая из маски краски не вырезается
    car_mask_keep_bottom: float = Field(default=0.25, ge=0, le=1)
    max_stroke: float = Field(default=0.15, gt=0)
    # склейка парных краёв: допуск по углу в градусах и по расстоянию между краями
    merge_angle_deg: float = Field(default=3.0, gt=0)
    merge_dist: float = Field(default=0.08, gt=0)
    # сшивка обрывков одной линии: минимальная длина обрывка в долях длинной стороны машины,
    # допуск по углу в градусах и наибольший разрыв
    piece_min: float = Field(default=0.06, ge=0)
    link_angle_deg: float = Field(default=5.0, gt=0)
    link_gap: float = Field(default=0.5, ge=0)
    # допустимая длина разделителя в долях локальной длинной стороны машины
    seg_len_min: float = Field(default=0.25, gt=0)
    seg_len_max: float = Field(default=2.0, gt=0)
    # сборка рядов: допуск по углу между разделителями в градусах, шаг между соседними
    # разделителями в долях ширины машины, коэффициент вариации шага, расстояние до соседнего
    # принятого ряда в долях длинной стороны машины
    row_angle_deg: float = Field(default=12.0, gt=0)
    gap_min: float = Field(default=0.5, gt=0)
    gap_max: float = Field(default=2.5, gt=0)
    # наибольший сдвиг соседних разделителей вдоль их направления в долях длинной стороны машины
    row_shift: float = Field(default=1.0, gt=0)
    # гибрид 7.2: ряд собирается от этого числа наблюдаемых разделителей
    row_min_dividers: int = Field(default=2, ge=2)
    # шаг ряда ровно из двух наблюдаемых разделителей в долях ширины машины: у пары нельзя
    # отличить одно место от двух, поэтому диапазон уже общего
    pair_gap_min: float = Field(default=0.7, gt=0)
    pair_gap_max: float = Field(default=1.4, gt=0)
    row_gap_cv: float = Field(default=0.35, gt=0)
    row_neighbour_gap: float = Field(default=2.5, gt=0)
    # наибольший пропуск, заполняемый интерполяцией, в медианных шагах ряда
    max_fill_gap: PositiveInt = 3
    # гибрид 7.2: ход ряда по сетке наружу прекращается после стольких не принятых ячеек подряд
    max_projected_run: PositiveInt = 2
    # допустимая площадь места в квадратах ширины машины
    slot_area_min: float = Field(default=0.6, gt=0)
    slot_area_max: float = Field(default=3.0, gt=0)
    # минимум мест для статуса ok
    min_slots: PositiveInt = 3


# настройки приложения: значения по умолчанию, затем YAML, затем .env, затем переменные окружения
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PARKINGSCAN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    models_dir: Path = Path("models")
    data_dir: Path = Path("data")
    max_upload_mb: PositiveInt = 50
    default_conf: float = Field(default=0.25, gt=0.0, le=1.0)
    # опорная точка машины по 7.3: насколько поднять середину нижней стороны бокса, в долях высоты
    ground_offset: float = Field(default=0.15, ge=0, lt=1)
    calibration: CalibrationSettings = CalibrationSettings()

    # путь к контракту инференса рядом с весами модели
    @property
    def inference_config_path(self) -> Path:
        return self.models_dir / "config.json"

    # порядок источников задаёт приоритет: первый в кортеже перекрывает остальные
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_FILE))
        yaml_settings = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file)
        return init_settings, env_settings, dotenv_settings, yaml_settings


# настройки читаются один раз за процесс
@lru_cache
def get_settings() -> Settings:
    return Settings()


# читаем и валидируем контракт инференса, при несовпадении схемы падаем сразу на старте
def load_inference_config(path: Path) -> InferenceConfig:
    return InferenceConfig.model_validate_json(path.read_text(encoding="utf-8"))
