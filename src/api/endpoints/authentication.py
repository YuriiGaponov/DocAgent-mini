"""
src.api.endpoints.authentication

Модуль эндпоинтов аутентификации для DocAgent-mini.

Предоставляет роутер с готовыми маршрутами для аутентификации и регистрации
пользователей, реализованными на базе fastapi-users.
Использует централизованную конфигурацию из src.users.auth_config
(менеджер пользователей, бэкенд аутентификации, схемы данных).
"""

from fastapi import APIRouter

from src.settings import settings
from src.users import auth_backend, fastapi_users, UserRead, UserCreate

"""Основной роутер аутентификации."""
router = APIRouter()

"""
Роутер для эндпоинтов аутентификации (логин/logout).

Использует auth_backend для выдачи токенов.
"""
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix=f'/{settings.AUTH_JWT_URL_PREFIX}',
    tags=["authentication"]
)

"""
Роутер для эндпоинтов регистрации пользователей.

Использует схемы UserCreate (вход) и UserRead (выход) для валидации данных.
"""
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix=f'/{settings.AUTH_JWT_URL_PREFIX}',
    tags=["register"]
)
