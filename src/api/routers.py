"""
src.api.routers

Модуль формирования главного роутера приложения DocAgent‑mini.

Централизует подключение всех групп маршрутов (роутеров) API, обеспечивая
единую точку сборки эндпоинтов для основного приложения.
"""

from fastapi import APIRouter

from src.api.endpoints import auth_router


"""Главный роутер приложения."""
router = APIRouter()

router.include_router(auth_router)
