"""
src.users.user_schemas

Модуль схем Pydantic для работы с пользователями в DocAgent-mini.

Определяет модели данных (Pydantic schemas) для сериализации и валидации
пользовательских объектов в API‑запросах и ответах. Основан на базовых схемах
из fastapi-users, что обеспечивает совместимость с системой аутентификации.
"""

from fastapi_users import schemas


class UserRead(schemas.BaseUser[int]):
    """
    Схема для чтения данных пользователя (ответ API).

    Наследует поля от BaseUser (fastapi-users) и специализирует ID как int
    (в соответствии с моделью User, использующей целочисленные идентификаторы).

    Включает базовые поля:
        - id (int): уникальный идентификатор пользователя;
        - email (str): адрес электронной почты;
        - is_active (bool): статус активности;
        - is_superuser (bool): флаг администратора;
        - is_verified (bool): статус верификации.
    """


class UserCreate(schemas.BaseUserCreate):
    """
    Схема для создания нового пользователя (запрос на регистрацию).

    Наследует обязательные поля от BaseUserCreate (fastapi-users):
        - email (str): адрес электронной почты (валидируется как email);
        - password (str): пароль (минимальная длина и сложность зависят
          от настроек fastapi-users).

    Опционально может включать:
        - is_active (bool): статус активности (по умолчанию true);
        - is_superuser (bool): флаг администратора (по умолчанию false);
        - is_verified (bool): статус верификации (по умолчанию false).
    """


class UserUpdate(schemas.BaseUserUpdate):
    """
    Схема для обновления данных пользователя (частичное обновление).

    Наследует логику от BaseUserUpdate (fastapi-users), где все поля
    опциональны.
    Позволяет изменять:
        - password (str | None): новый пароль (если передан);
        - is_active (bool | None): новый статус активности;
        - is_superuser (bool | None): новый флаг администратора;
        - is_verified (bool | None): новый статус верификации.
    """
