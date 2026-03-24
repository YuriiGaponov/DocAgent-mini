"""
Модуль main.py — точка входа в приложение DocAgent‑mini.

Инициализирует FastAPI‑приложение с конфигурацией из настроек проекта,
обеспечивающее HTTP‑интерфейс для взаимодействия с AI‑агентом
по работе с внутренней документацией компании.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src.logger import logger
from src.api import router
from src.settings import Settings, get_settings


logger.warning('НАЧАЛО РАБОТЫ DocAgent‑mini')


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Менеджер жизненного цикла приложения (lifespan).

    Обеспечивает корректное управление ресурсами при запуске
    и завершении работы приложения. Выполняет логирование ключевых
    этапов жизненного цикла и базовую обработку исключений.
    """
    logger.info('Начало запуска DocAgent‑mini')
    try:
        logger.success('DocAgent‑mini успешно запущен')
        yield
    except Exception as e:
        logger.critical(f'Ошибка запуска DocAgent‑mini: {e}')
        raise
    finally:
        logger.warning('ЗАВЕРШЕНИЕ РАБОТЫ DocAgent‑mini')

settings: Settings = get_settings()
"""Настройки приложения, загруженные через get_settings()."""

app = FastAPI(
    debug=settings.DEBUG,
    title="DocAgent‑mini",
    description="AI‑агент для работы с внутренней документацией компании",
    lifespan=lifespan
)
"""Экземпляр FastAPI — веб‑сервер приложения."""

logger.debug(f'Приложение DocAgent‑mini инициализировано: {app.__class__}')

app.include_router(router)
logger.debug(f'Подключен роутер: {router.__class__}')
