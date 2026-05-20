"""
src.settings

Модуль settings проекта DocAgent‑mini.

Содержит основные настройки и конфигурационные параметры,
используемые в проекте.

Модуль обеспечивает централизованное управление конфигурацией,
упрощает внесение изменений и поддерживает разделение настроек
для разных окружений (разработка, тестирование, продакшен).
"""


import os
from pathlib import Path
from typing import Literal


from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
"""Корневой путь проекта — директория, расположенная на уровень выше src."""

ENV_MODE = os.getenv('ENVIRONMENT', 'development')
"""Режим окружения: из ENVIRONMENT или 'development' по умолчанию."""


class Settings(BaseSettings):
    """
    Класс для централизованного управления настройками проекта.
    Считывает параметры из файла .env и переменных окружения.
    """

    # === Описание ===
    TITLE: str = 'DocAgent-mini'
    """str: Название приложения."""

    DESCRIPTION: str = (
        'Мини‑проект AI‑агента, который отвечает на вопросы пользователей '
        'по внутренней документации.'
    )
    """str: Краткое описание приложения."""

    VERSION: str = '2.0'
    """str: Версия приложения."""

    # === Кодировка ===
    ENCODING: str = 'UTF‑8'
    """str: Кодировка для обработки текстовых данных."""

    # === Режим отладки ===
    DEBUG: bool = False
    """bool: Флаг режима отладки приложения. По умолчанию — False."""

    # === Настройки логирования ===
    LOG_DIR: Path = BASE_DIR / 'logs'
    """Path: Директория для логов приложения. По умолчанию — 'logs'."""

    LOG_FILENAME: str = 'app_log.log'
    """str: Имя файла логов. По умолчанию — 'app_log.log'."""

    @computed_field
    @property
    def LOG_FILE(self) -> Path:
        """
        Формирует полное имя файла логов из LOG_DIR и LOG_FILENAME.

        Returns:
            Path: Полный путь к файлу логов.
        """
        return self.LOG_DIR / self.LOG_FILENAME

    MIN_LOG_LEVEL: Literal[
        'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    ] = 'INFO'
    """str: Минимальный уровень логирования. По умолчанию — 'INFO'."""

    @computed_field
    @property
    def LOG_LEVEL(self) -> str:
        """
        Устанавливает минимальный уровень логирования.

        Если режим отладки включен — 'DEBUG'.
        Иначе устанавливается значение MIN_LOG_LEVEL.

        Returns:
            str: Уровень логирования.
        """
        if self.DEBUG:
            return 'DEBUG'
        return self.MIN_LOG_LEVEL

    # === Реляционная база данных ===
    DBMS: Literal['sqlite'] = 'sqlite'
    """str: СУБД реляционной базы данных."""

    DB_DIR: Path = BASE_DIR / 'data'
    """Path: Директория для Реляционной БД. По умолчанию — 'data'."""

    DB_NAME: str = 'DocAgent-mini.db'
    """str: Имя БД. По умолчанию — 'app_log.log'."""

    @computed_field
    @property
    def DB(self) -> Path:
        """
        Формирует полное имя БД из DB_DIR и DB_NAME.

        Returns:
            Path: Полный путь к БД.
        """
        return self.DB_DIR / self.DB_NAME

    # Конфигурация получения настроек
    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / f".env.{ENV_MODE}"),
        env_file_encoding='utf-8',
        extra='ignore'
    )


_settings: Settings = Settings()
"""Глобальный экземпляр настроек."""


def get_settings() -> Settings:
    """
    Возвращает глобальный экземпляр настроек приложения.

    Выступает как единый интерфейс для получения настроек.
    При тестировании легко заменяется на мок.

    Returns:
        Settings: Настроенный экземпляр класса настроек приложения.
    """
    return _settings
