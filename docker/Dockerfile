# syntax=docker/dockerfile:1

# Образ под CPU. Сборка из корня репозитория:
#   docker build -f docker/Dockerfile -t parkingscan .

# версия Python та же, что в локальном окружении
ARG PYTHON_VERSION=3.14

# сборочная стадия: окружение в /opt/venv, которое целиком копируется в итоговый образ
FROM python:${PYTHON_VERSION}-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
# версии torch и torchvision должны совпадать с uv.lock, OpenCV - с pyproject.toml
ARG TORCH_VERSION=2.14.0
ARG TORCHVISION_VERSION=0.29.0
ARG OPENCV_VERSION=5.0.0.93
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
# pyproject.toml привязывает torch к индексу cu130 и для Linux, в CPU-образ это не годится:
# зависимости экспортируются из uv.lock без torch и его CUDA-библиотек,
# а torch ставится с CPU-индекса PyTorch
RUN uv venv /opt/venv \
    && uv export --frozen --no-dev --no-emit-project --no-hashes \
        --prune torch --prune torchvision --output-file /tmp/requirements.txt \
    && uv pip install --python /opt/venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    && uv pip install --python /opt/venv/bin/python --requirement /tmp/requirements.txt \
    && uv pip install --python /opt/venv/bin/python --no-deps .
# ultralytics тянет opencv-python, которому нужны libGL и X11, а в образе без экрана их нет.
# Headless-сборку ultralytics не выпускает и рекомендует заменять пакет после установки:
# оба пакета дают один модуль cv2, поэтому удаляются оба и ставится только headless той же версии
RUN uv pip uninstall --python /opt/venv/bin/python opencv-python opencv-python-headless \
    && uv pip install --python /opt/venv/bin/python --no-deps opencv-python-headless==${OPENCV_VERSION}


# итоговый образ: окружение, настройки и веса модели; данные монтируются томом в /app/data
FROM python:${PYTHON_VERSION}-slim
# настройки ultralytics пишутся в том с данными, а не в домашний каталог
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/app/data/ultralytics
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY configs ./configs
COPY models ./models
# сервис работает от непривилегированного пользователя; каталог данных принадлежит ему,
# чтобы запись работала и без тома, и с именованным томом
RUN useradd --create-home --uid 1000 parkingscan \
    && mkdir -p /app/data \
    && chown parkingscan:parkingscan /app/data
USER parkingscan
EXPOSE 8000
CMD ["uvicorn", "parkingscan.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
