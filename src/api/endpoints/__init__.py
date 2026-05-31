"""
src.api.endpoints.__init__

Пакет эндпоинтов API для проекта DocAgent‑mini.

Централизует экспорт роутеров, сгруппированных по выполняемым функциям.
"""

from src.api.endpoints.authentication import router as auth_router

__all__ = ['auth_router']
