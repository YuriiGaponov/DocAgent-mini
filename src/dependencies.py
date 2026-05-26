"""
src.dependencies

Модуль зависимостей для приложения DocAgent-mini.

Предоставляет готовые зависимости (в терминах FastAPI) для внедрения
в эндпоинты, прежде всего — для работы с БД. Централизует создание
зависимостей, чтобы избежать дублирования логики в роутах.
"""

from src.db import providerDB, create_async_session_dependency
from src.users import get_user_db_dependency


"""Зависимость для получения асинхронной сессии БД."""
async_session_dependency = create_async_session_dependency(providerDB)

"""Зависимость для получения хранилища пользователей в FastAPI."""
user_db_dependency = get_user_db_dependency(async_session_dependency)
