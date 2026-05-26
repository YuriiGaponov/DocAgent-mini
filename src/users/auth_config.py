"""
src.users.auth_config

Модуль конфигурации аутентификации для DocAgent-mini.

Определяет компоненты аутентификации на основе fastapi-users:
- транспорт Bearer (BearerTransport) для передачи JWT-токенов;
- стратегию JWT (JWTStrategy) с настройками из конфига;
- бэкенд аутентификации (AuthenticationBackend),
    объединяющий транспорт и стратегию.
"""


from typing import Callable

from sqlalchemy.ext.asyncio import AsyncSession
from fastapi_users import BaseUserManager, FastAPIUsers, IntegerIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend, BearerTransport, JWTStrategy
)
from fastapi_users.db import SQLAlchemyUserDatabase

from src.db import User
from src.settings import settings


async def get_user_db_dependency(
    session_dependency: Callable[[], AsyncSession]
):
    """
    Зависимость для получения экземпляра базы данных пользователей.

    Args:
        session_dependency: Функция, возвращающая зависимость сессии БД.
    """
    session = await session_dependency()
    try:
        yield SQLAlchemyUserDatabase(session, User)
    finally:
        await session.close()


"""
Транспорт для аутентификации по схеме Bearer.

Используется для передачи JWT‑токенов в заголовках HTTP‑запросов.
Определяет endpoint для получения токена (tokenUrl).
"""
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    """
    Фабрика для создания стратегии аутентификации JWT.

    Использует секретный ключ и время жизни токена из настроек приложения.

    Returns:
        JWTStrategy: стратегия аутентификации, готовая к использованию
                     в AuthenticationBackend.
    """
    return JWTStrategy(
        secret=settings.JWT_KEY_SECRET,
        lifetime_seconds=settings.JWT_TOKEN_LIFETIME
    )


"""
Бэкенд аутентификации для системы пользователей.

Объединяет транспорт (BearerTransport) и стратегию (JWTStrategy),
чтобы обеспечить полный цикл аутентификации в fastapi-users.
"""
auth_backend = AuthenticationBackend(
    name='DocAgent-mini_auth_backend',
    transport=bearer_transport,
    get_strategy=get_jwt_strategy
)


class UserManager(BaseUserManager, IntegerIDMixin):
    """
    Менеджер пользователей для системы аутентификации DocAgent-mini.

    Наследует базовую функциональность от BaseUserManager (fastapi-users)
    и реализует поддержку целочисленных ID через IntegerIDMixin.

    Отвечает за бизнес‑логику операций с пользователями:
    - создание учётных записей;
    - верификация email;
    - сброс пароля;
    - управление статусом учётной записи (активация, блокировка);
    - генерация и валидация токенов для действий
      (например, подтверждения email).

    Интегрируется с:
    - auth_backend: для реализации механизмов аутентификации;
    - БД через модели (например, User): для сохранения и извлечения данных;
    - транспортом (BearerTransport): для выдачи JWT‑токенов после успешной
      аутентификации.
    """


async def get_user_manager_dependency(
    user_db_dependency: Callable[[], AsyncSession]
):
    """
    Зависимость для получения экземпляра менеджера пользователей в FastAPI.

    Args:
        user_db_dependency: Функция‑зависимость,
            возвращающая экземпляр SQLAlchemyUserDatabase.
    """
    user_db = await user_db_dependency()
    yield UserManager(user_db)


def get_fastapi_users(user_manager: Callable) -> FastAPIUsers:
    """
    Фабрика для создания экземпляра FastAPIUsers — центрального компонента
    системы аутентификации.

    Объединяет менеджер пользователей и бэкенды аутентификации в единый объект,
    который предоставляет готовые эндпоинты для работы с пользователями
    """
    return FastAPIUsers(
        get_user_manager=user_manager,
        auth_backends=[auth_backend]
    )
