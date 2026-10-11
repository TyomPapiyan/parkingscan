from typing import Annotated

import cv2
import numpy as np
from fastapi import Depends, HTTPException, Request, UploadFile, status

from parkingscan.api.schemas import DeviceInfo
from parkingscan.config import Settings, get_settings
from parkingscan.core.detector import CarDetector


# устройство инференса, выбранное один раз при старте приложения
def get_device(request: Request) -> DeviceInfo:
    return request.app.state.device


# детектор машин, загруженный один раз при старте приложения
def get_detector(request: Request) -> CarDetector:
    return request.app.state.detector


DeviceDep = Annotated[DeviceInfo, Depends(get_device)]
DetectorDep = Annotated[CarDetector, Depends(get_detector)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# общий для роутеров разбор загрузки: ограничение размера и декодирование в RGB
def read_image(upload: UploadFile, max_upload_mb: int) -> np.ndarray:
    max_bytes = max_upload_mb * 1024 * 1024
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"File exceeds {max_upload_mb} MB")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is not a decodable image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
