import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from parkingscan.api.routes_occupancy import DEFAULT_RENDER_SIDE, analyse, build_response, draw
from parkingscan.config import get_settings, load_inference_config
from parkingscan.core.detector import MODEL_CONF, CarDetector, select_device
from parkingscan.core.render import encode_image
from parkingscan.logging_conf import setup_logging

# расширения файлов, которые считаются снимками при обходе папки
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# порог уверенности из командной строки в тех же границах, что и в API
def conf_threshold(value: str) -> float:
    conf = float(value)
    if not MODEL_CONF <= conf <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between {MODEL_CONF} and 1.0")
    return conf


# один файл либо все снимки папки по алфавиту
def collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [path]


# снимок в RGB; чтение через буфер, потому что cv2.imread на Windows не открывает пути с кириллицей
def read_rgb(path: Path) -> np.ndarray | None:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# батч-режим без API, разовый режим 7.2: на каждый снимок в stdout печатается строка JSON
# с ответом раздела 7.3, логи идут в stderr; с --out рядом сохраняется отрисовка
def main() -> None:
    parser = argparse.ArgumentParser(prog="parkingscan", description="Parking occupancy for an image or a folder")
    parser.add_argument("path", type=Path, help="image file or folder with images")
    parser.add_argument("--conf", type=conf_threshold, help="confidence threshold, defaults to default_conf")
    parser.add_argument("--out", type=Path, help="folder for rendered JPEG images")
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"{args.path} does not exist")
    images = collect_images(args.path)
    if not images:
        parser.error(f"no images in {args.path}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    setup_logging(settings.log_level)
    config = load_inference_config(settings.inference_config_path)
    detector = CarDetector(settings.models_dir / config.model_file, config, select_device()[0])

    failed = 0
    for path in images:
        image = read_rgb(path)
        if image is None:
            print(f"{path}: not a decodable image", file=sys.stderr)
            failed += 1
            continue
        result = analyse(image, detector, settings, args.conf, None)
        print(build_response(result).model_dump_json())
        if args.out:
            rendered = encode_image(draw(image, result, DEFAULT_RENDER_SIDE), "jpeg")
            (args.out / f"{path.stem}_occupancy.jpg").write_bytes(rendered)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
