"""
Модуль src.api.py — определение API‑эндпоинтов
приложения DocAgent‑mini.

Содержит маршрутизатор FastAPI (APIRouter) и базовые
эндпоинты, включая проверку работоспособности
сервиса (/health).
"""

import time

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
    logger.info('Начало обработки запроса на эндпоинт "/health"')
    try:
        logger.success('Статус приложения: "healthy"')
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f'Ошибка при проверке здоровья приложения: {e}')
        return {"status": "unhealthy", "error": str(e)}


@router.post("/ask")
async def ask(request_data: AskRequest):
    pass


# Временный функционал для целей разработки.
# Будет удален из релизной версии.
from fastapi import Depends
from src.settings import Settings, get_settings
from src.rag import RAGSystem

@router.get("/")
async def temp_endpoint(settings: Settings = Depends(get_settings)):
    """
    Временный эндпоинт для тестирования функционала.
    Будет удален из релизной версии.
    """
    start_time = time.time()  # Замер времени начала
    logger.info('Начало обработки запроса на эндпоинт "/"')
    logger.debug(f'Получен экземпляр настроек: {settings.__class__}')
    rag_sys = RAGSystem(settings)
    logger.debug(f'Создан экземпляр RAG-системы: {rag_sys.__class__}')
    result = await rag_sys.create_docs_collection()
    end_time = time.time()  # Замер времени окончания
    execution_time = end_time - start_time  # Расчёт времени выполнения
    logger.info(f'Время выполнения temp_endpoint: {execution_time:.4f} секунд')
    return result
