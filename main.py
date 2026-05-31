"""
main.py

Основной модуль приложения DocAgent‑mini.

Инициализирует экземпляр FastAPI‑приложения с настройками, загруженными
из модуля src. Использует конфигурационные параметры для настройки режима
отладки, заголовка, описания и версии API. Настраивает жизненный цикл
приложения с асинхронным контекстом для логирования ключевых этапов работы
(запуск, ошибки, завершение).

"""


from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from src import get_admin, router, settings, tags_metadata
from src.logger import app_logger as logger


logger.warning('НАЧАЛО РАБОТЫ')


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Асинхронный контекст для управления жизненным циклом FastAPI-приложения.

    Логирует начало запуска приложения, фиксирует успешный запуск,
    обрабатывает критические ошибки и логирует завершение работы.

    Args:
        app (FastAPI): экземпляр приложения FastAPI.

    Yields:
        None: управление передаётся основному циклу приложения.

    Raises:
        Exception: при возникновении ошибки во время работы приложения —
                   ошибка логируется и пробрасывается дальше.
    """
    logger.info(f'начало запуска {app.title}')
    try:
        logger.info(f'успешный запуск {app.title}')
        yield
    except Exception as e:
        logger.critical(
            f'ошибка запуска {app.title}',
            extra={'error': str(e)}
        )
        raise
    finally:
        logger.warning('ЗАВЕРШЕНИЕ РАБОТЫ')


app = FastAPI(
    debug=settings.DEBUG,
    title=settings.TITLE,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    openapi_tags=tags_metadata,
    lifespan=lifespan
)

logger.debug(f'приложение {app.title} инициализировано')

app.include_router(router)
logger.debug('подключен роутер')

admin = get_admin(app)
logger.debug('подключена роуадмин-панель')
