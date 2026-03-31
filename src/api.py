"""
Модуль src.api.py — определение API‑эндпоинтов
приложения DocAgent‑mini.

Содержит маршрутизатор FastAPI (APIRouter) и базовые
эндпоинты, включая проверку работоспособности
сервиса (/health).
"""

from fastapi import APIRouter

from src.logger import logger
from src.models import AskRequest

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
    logger.info('Запрос на эндпоинт "/health"')
    try:
        logger.success('Статус приложения: "healthy"')
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f'Ошибка при проверке здоровья приложения: {e}')
        return {"status": "unhealthy", "error": str(e)}


@router.post("/ask")
async def ask(request_data: AskRequest):
    logger.info('Запрос на эндпоинт "/ask"')
    pass
