import logging.config

# формат строки лога: время, уровень, имя логгера, сообщение
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


# единый вывод в консоль для приложения и uvicorn
def setup_logging(level: str) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {"format": LOG_FORMAT}},
            "handlers": {
                "console": {"class": "logging.StreamHandler", "formatter": "default"},
            },
            "root": {"level": level, "handlers": ["console"]},
            # у логгеров uvicorn снимаем собственные обработчики, чтобы они писали через корневой
            "loggers": {
                "uvicorn": {"handlers": [], "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": True},
            },
        }
    )
