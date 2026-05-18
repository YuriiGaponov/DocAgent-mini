"""
src.logger

Модуль для настройки логирования в проекте DocAgent‑mini.

Предоставляет функцию для создания настроенного экземпляра логгера,
который записывает сообщения в JSON‑формате с ротацией файлов.
"""

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler

from pythonjsonlogger import jsonlogger

from src.settings import Settings, get_settings


settings: Settings = get_settings()


def get_logger(name: str, settings: Settings) -> Logger:
    """
    Создаёт и настраивает экземпляр логгера с ротацией файлов
    и JSON‑форматированием.

    Устанавливает уровень логирования из настроек, добавляет обработчик
    с ротацией файлов и форматированием записей в JSON.

    Args:
        name (str): Имя логгера.
        settings (Settings): Экземпляр настроек приложения.

    Returns:
        Logger: Настроенный экземпляр логгера.
    """
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)

    handler = RotatingFileHandler(
        filename=settings.LOG_FILE_PATH,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding=settings.ENCODING
    )

    formatter = jsonlogger.JsonFormatter(
        (
            "%(asctime)s %(name)s %(filename)s %(lineno)d %(levelname)s"
            "%(message)s"
        ),
        json_ensure_ascii=False
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


app_logger = get_logger('app_logger', settings)
"""
Экземпляр логгера для основного приложения.
Использует настройки из модуля settings и форматирует логи в JSON.
"""
