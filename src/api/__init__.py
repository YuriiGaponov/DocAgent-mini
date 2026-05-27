"""
src.api.__init__

Пакет API для проекта DocAgent-mini.

Централизует доступ к роутерам API, обеспечивая удобную интеграцию с основным
приложением.
"""

from src.api.routers import router

__all__ = ['router']
