"""
src.users.__init__

Пакет управления пользователями в проекте DocAgent-mini.

Предоставляет компоненты для работы с пользовательскими учётными записями:
- аутентификация и авторизация;
- CRUD-операции над пользователями;
- интеграция с системой БД (через модели из src.db.models);
- взаимодействие с FastAPI для реализации эндпоинтов.
"""

from src.users.auth_config import (
    auth_backend, get_user_db, get_user_manager,
    fastapi_users
)
from src.users.user_schemas import UserCreate, UserRead

__all__ = [
    'auth_backend', 'fastapi_users', 'get_user_db', 'get_user_manager',
    'UserCreate', 'UserRead'
]
