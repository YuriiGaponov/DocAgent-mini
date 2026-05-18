"""
src.settings

Модуль settings проекта DocAgent‑mini.

Содержит основные настройки и конфигурационные параметры,
используемые в проекте.

Модуль обеспечивает централизованное управление конфигурацией,
упрощает внесение изменений и поддерживает разделение настроек
для разных окружений (разработка, тестирование, продакшен).

Version: 2.0
"""


from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
"""Корневой путь проекта — директория, расположенная на уровень выше src."""


class Settings(BaseSettings):
    """
    Класс для централизованного управления настройками проекта.
    Считывает параметры из файла .env и переменных окружения.
    """
    # Описание
    DESCRIPTION: str = (
        'мини‑проект AI‑агента, который отвечает на вопросы пользователей '
        'по внутренней документации'
    )
    # Режим отладки
    DEBUG: bool = False
    """bool: Флаг режима отладки приложения. По умолчанию — False."""

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
