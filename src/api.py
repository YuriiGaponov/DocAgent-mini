"""
Модуль src.api.py — определение API‑эндпоинтов
приложения DocAgent‑mini.

Содержит маршрутизатор FastAPI (APIRouter) и базовые
эндпоинты, включая проверку работоспособности
сервиса (/health).
"""

from fastapi import APIRouter

router = APIRouter()
"""Экземпляр APIRouter — маршрутизатор
для группировки эндпоинтов API."""


@router.get("/health")
async def health():
    """
    Эндпоинт проверки работоспособности приложения
    (health check).

    Возвращает статус доступности сервиса. В случае
    успешного выполнения сообщает о работоспособности
    системы, при возникновении ошибок — указывает
    на проблемы.

    Returns:
        dict: JSON‑ответ со статусом сервиса:
            * При успехе: {"status": "healthy"}
            * При ошибке: {"status": "unhealthy",
              "error": <сообщение об ошибке>}

    HTTP‑статус:
        Всегда возвращает 200 OK, даже в случае
        внутренних ошибок, чтобы чётко различать
        доступность сервиса и его внутреннее состояние.
    """
    try:
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# Временный функционал для целей разработки.
# Будет удален из релизной версии.
from fastapi import Depends
from src.settings import Settings, get_settings
from src.tools import get_docs

@router.get("/")
async def temp_endpoint(settings: Settings = Depends(get_settings)):
    """
    Временный эндпоинт для тестирования функционала.
    Будет удален из релизной версии.
    """
    return await get_docs(settings)
