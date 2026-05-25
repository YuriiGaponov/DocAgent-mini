"""
src.db.models

Модуль моделей реляционной базы данных для DocAgent-mini.

Определяет классы моделей, наследуемые от Base (из src.db.relational_db),
для хранения структурированных данных в БД:
- `User`: модель пользователя.
"""

from fastapi_users.db import SQLAlchemyBaseUserTable

from src.db.relational_db import Base


class User(SQLAlchemyBaseUserTable, Base):
    """
    Модель пользователя для системы аутентификации и авторизации.

    Наследует поля и поведение от SQLAlchemyBaseUserTableUUID (fastapi-users)
    и подключается к общей схеме БД через наследование от Base.

    Автоматически получает:
    - __tablename__ = 'user' (формируется из имени класса через Base);
    - интеграцию с асинхронным движком и сессиями;
    - поле id (Integer, первичный ключ) от Base.

    Поля, унаследованные от SQLAlchemyBaseUserTable:
        - email (str): адрес электронной почты (уникальный ключ);
        - hashed_password (str): хешированный пароль;
        - is_active (bool): флаг активности учётной записи;
        - is_superuser (bool): флаг администратора;
        - is_verified (bool): флаг подтверждённой учётной записи.

    Note:
        Для добавления кастомных полей (например, full_name, department)
        достаточно объявить их как Column внутри класса.
    """
    pass
