"""
Модуль main.py — точка входа в приложение DocAgent‑mini.

Инициализирует FastAPI‑приложение с конфигурацией из настроек проекта,
обеспечивающее HTTP‑интерфейс для взаимодействия с AI‑агентом
по работе с внутренней документацией компании.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.settings import Settings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Менеджер жизненного цикла приложения (lifespan).

    Обеспечивает корректное управление ресурсами при запуске
    и завершении работы приложения. В текущей реализации
    просто передаёт управление дальше (yield), с базовой обработкой исключений.
    """
    try:
        yield
    except Exception:
        raise

settings: Settings = get_settings()
"""Настройки приложения, загруженные через get_settings()."""

app = FastAPI(
    debug=settings.DEBUG,
    title="DocAgent‑mini",
    description="AI‑агент для работы с внутренней документацией компании"
)
"""Экземпляр FastAPI — веб‑сервер приложения."""
