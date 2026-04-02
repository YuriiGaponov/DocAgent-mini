"""
Модуль src.settings.py — конфигурация приложения DocAgent‑mini.

Определяет класс настроек Settings на базе Pydantic BaseSettings для загрузки
параметров из файла .env. Предоставляет функцию get_settings() для получения
экземпляра настроек с учётом переменных окружения и значений по умолчанию.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
"""Корневой путь проекта — директория, расположенная на уровень выше src."""


class Settings(BaseSettings):
    """
    Класс настроек приложения, наследующий BaseSettings из Pydantic.

    Загружает параметры конфигурации из файла .env с возможностью
    переопределения через переменные окружения. Содержит настройки
    режима отладки и логирования.
    """

    # Режим отладки
    DEBUG: bool = False
    """bool: Флаг режима отладки приложения. По умолчанию — False."""

    # Настройки логирования
    LOG_DIR: str = 'logs'
    """str: Директория для хранения логов приложения. По умолчанию — 'logs'."""

    LOG_FILENAME: str = 'app_log.log'
    """str: Имя файла логов. По умолчанию — 'app_log.log'."""

    # Настройки RAG-системы
    DOC_PATH: str = f'{BASE_DIR}/docs'
    ALLOWED_FILENAME_PATTERN: str = r"^[a-z0-9_-]+\.md$"
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    VECTOR_DB_NAME: str = 'docs'

    # Настройки LLM
    LLM_MODEL: str = 'mistral'

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / '.env'),
        env_file_encoding='utf-8',
        extra='ignore'
    )


def get_settings() -> Settings:
    """
    Возвращает экземпляр класса Settings с загруженными настройками.

    Создаёт новый объект Settings, который автоматически считывает
    параметры из файла .env и переменных окружения согласно
    конфигурации model_config.

    Returns:
        Settings: Настроенный экземпляр класса настроек приложения.
    """
    return Settings()
