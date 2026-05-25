"""
src.users.users_config

Модуль конфигурации аутентификации для DocAgent-mini.

Определяет компоненты аутентификации на основе fastapi-users:
- транспорт Bearer (BearerTransport) для передачи JWT-токенов;
- стратегию JWT (JWTStrategy) с настройками из конфига;
- бэкенд аутентификации (AuthenticationBackend),
    объединяющий транспорт и стратегию.
"""

from fastapi_users.authentication import (
    AuthenticationBackend, BearerTransport, JWTStrategy
)

from src.settings import settings


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
