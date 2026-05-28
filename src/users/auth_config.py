"""
src.users.auth_config

Модуль конфигурации аутентификации для DocAgent-mini.

Определяет компоненты аутентификации на основе fastapi-users:
- транспорт Bearer (BearerTransport) для передачи JWT-токенов;
- стратегию JWT (JWTStrategy) с настройками из конфига;
- бэкенд аутентификации (AuthenticationBackend),
    объединяющий транспорт и стратегию.
"""


from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends
from fastapi_users import BaseUserManager, FastAPIUsers, IntegerIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend, BearerTransport, JWTStrategy
)
from fastapi_users.db import SQLAlchemyUserDatabase

from src.db import User, async_session_dependency
from src.settings import settings


async def get_user_db(
        session: AsyncSession = Depends(async_session_dependency)
):
    """
    Зависимость для получения экземпляра адаптера базы данных пользователей.

    Args:
        session (AsyncSession): активная асинхронная сессия SQLAlchemy,
            предоставляемая через зависимость async_session_dependency.

    Yields:
        SQLAlchemyUserDatabase: адаптер для выполнения CRUD‑операций
            над записями пользователей в БД.
    """
    yield SQLAlchemyUserDatabase(session, User)

"""
Транспорт для аутентификации по схеме Bearer.

Используется для передачи JWT‑токенов в заголовках HTTP‑запросов.
Определяет endpoint для получения токена (tokenUrl).
"""
bearer_transport = BearerTransport(
    tokenUrl=f"{settings.AUTH_JWT_URL_PREFIX}/login"
)


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


async def get_user_manager(user_db=Depends(get_user_db)):
    """
    Зависимость для получения экземпляра менеджера пользователей.

    Args:
        user_db (SQLAlchemyUserDatabase): адаптер БД, предоставляемый
            через зависимость get_user_db.

    Yields:
        UserManager: менеджер пользователей, готовый к работе
            с учётными записями (регистрация, обновление, верификация и т. д.).
    """
    yield UserManager(user_db)


"""
Экземпляр FastAPIUsers — центральный компонент системы управления
пользователями.

Предоставляет набор готовых роутеров для интеграции в приложение FastAPI:
- auth_router: эндпоинты аутентификации (логин, logout);
- register_router: регистрация новых пользователей;
- reset_password_router: восстановление пароля;
- verify_email_router: подтверждение email;
- users_router: управление учётными записями.
"""
fastapi_users = FastAPIUsers(
    get_user_manager,
    [auth_backend],
)
