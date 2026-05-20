"""
src.db.relational_db

Модуль работы с реляционной базой данных в проекте DocAgent‑mini.

Реализует инфраструктуру для взаимодействия с реляционным хранилищем
на базе SQLAlchemy с поддержкой асинхронности:
- определяет базовый класс моделей с предустановленными общими атрибутами;
- настраивает подключение к БД с учётом выбранного типа СУБД;
- создаёт асинхронный движок (engine) для выполнения операций с данными.

Ключевые компоненты модуля:

- `PreBase`: базовый класс для моделей БД. Автоматически:
  - формирует имя таблицы на основе имени класса (через `__tablename__`);
  - добавляет поле `id` как первичный ключ.
- `Base`: декларативная база SQLAlchemy, наследующая функционал от `PreBase`.
    Служит родительским классом для всех моделей реляционной БД в проекте.
- `get_db_url()`: функция для формирования URL подключения к БД
    на основе настроек проекта. Поддерживает расширение за счёт добавления
    новых СУБД в словарь `available_db_url`.
- `engine`: асинхронный движок SQLAlchemy, инициализированный
    с URL подключения.

Цель модуля — предоставить унифицированный и расширяемый слой доступа
к реляционной БД для хранения структурированных данных
(задач, истории взаимодействий и т. д.) в рамках AI‑агента.
"""

from sqlalchemy import Column, Integer
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declared_attr, declarative_base

from src.settings import get_settings, Settings


class PreBase:
    """
    Базовый класс с предустановленными параметрами для моделей БД.

    Обеспечивает автоматическое формирование имени таблицы
    на основе имени класса и добавляет поле `id`.
    """

    @declared_attr
    def __tablename__(cls):
        """
        Формирует имя таблицы как строчное представление имени класса.
        """
        return cls.__name__.lower()

    id = Column(Integer, primary_key=True)


Base = declarative_base(cls=PreBase)
"""
Декларативная база для моделей SQLAlchemy.

Наследует параметры от `PreBase`, включая автоматическое имя таблицы
и поле `id`. Используется как родительский класс для всех моделей БД.
"""

settings = get_settings()


def get_db_url(settings: Settings) -> str:
    """
    Формирует URL подключения к базе данных на основе настроек проекта.

    Args:
        settings (Settings): объект настроек приложения.

    Returns:
        str: URL для подключения к выбранной СУБД.

    Примечание:
        Словарь `available_db_url` можно расширять для поддержки новых СУБД.
    """
    available_db_url: dict = {
        'sqlite': f'sqlite+aiosqlite:///{settings.DB}'
    }
    return available_db_url[settings.DBMS]


DATABASE_URL = get_db_url(settings)

engine = create_async_engine(DATABASE_URL)
