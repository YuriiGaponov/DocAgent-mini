"""
src.users.__init__

Пакет управления пользователями в проекте DocAgent-mini.

Предоставляет компоненты для работы с пользовательскими учётными записями:
- аутентификация и авторизация;
- CRUD-операции над пользователями;
- интеграция с системой БД (через модели из src.db.models);
- взаимодействие с FastAPI для реализации эндпоинтов.
"""

from src.db import async_session_dependency
from src.users.auth_config import (
    get_user_db_dependency, get_user_manager_dependency, get_fastapi_users
)

"""Зависимость для получения хранилища пользователей в FastAPI."""
user_db_dependency = get_user_db_dependency(async_session_dependency)

"""Зависимость для получения экземпляра менеджера пользователей в FastAPI."""
user_manager_dependency = get_user_manager_dependency(user_db_dependency)

"""Центральный компонент системы аутентификации и управления пользователями."""
fastapi_users = get_fastapi_users(user_manager_dependency)

__all__ = ['fastapi_users']
