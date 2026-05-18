"""
main.py

Основной модуль приложения DocAgent‑mini версии 2.0.

Инициализирует экземпляр FastAPI-приложения с настройками, загруженными
из модуля src. Использует конфигурационные параметры для настройки режима
отладки и описания API.

Version: 2.0
"""

from fastapi import FastAPI

from src import get_settings

settings = get_settings()

app = FastAPI(
    debug=settings.DEBUG,
    description=settings.DESCRIPTION
)
