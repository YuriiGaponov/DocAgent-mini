"""
src.db.relational_db

Модуль работы с реляционной базой данных в проекте DocAgent‑mini.

Реализует инфраструктуру для взаимодействия с реляционным хранилищем
на базе SQLAlchemy с поддержкой асинхронности:
- определяет базовый класс моделей с предустановленными общими атрибутами;
- настраивает подключение к БД с учётом выбранного типа СУБД;
- создаёт асинхронный движок (engine) для выполнения операций с данными.

Ключевые компоненты модуля:

- `PreBase`: базовый класс для моделей БД.
- `Base`: декларативная база SQLAlchemy.
- `get_db_url()`: функция для формирования URL подключения к БД.
- `async_engine`: асинхронный движок SQLAlchemy.
- `get_async_session()`:  асинхронный контекстный менеджер сессии БД.

Цель модуля — предоставить унифицированный и расширяемый слой доступа
к реляционной БД для хранения структурированных данных
(задач, истории взаимодействий и т. д.) в рамках AI‑агента.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import Column, Integer
from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
)
from sqlalchemy.orm import declared_attr, declarative_base

from src.settings import Settings, get_settings


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

settings: Settings = get_settings()
"""Экземпляр настроек приложения."""


def get_db_url() -> str:
    """
    Формирует URL подключения к базе данных на основе настроек проекта.

    Returns:
        str: URL для подключения к выбранной СУБД.

    Примечание:
        Словарь `available_db_url` можно расширять для поддержки новых СУБД.
    """
    available_db_url: dict = {
        'sqlite': f'sqlite+aiosqlite:///{settings.DB}'
    }
    return available_db_url[settings.DBMS]


async_engine: AsyncEngine = create_async_engine(get_db_url(settings))
"""Асинхронный движок SQLAlchemy для выполнения запросов к БД."""

async_session_maker = async_sessionmaker(async_engine, expire_on_commit=False)
"""Фабрика асинхронных сессий."""


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Асинхронный контекстный менеджер для получения сессии БД.

    Создаёт и предоставляет асинхронную сессию в контексте async with.

    Yields:
        AsyncSession: активная асинхронная сессия SQLAlchemy для выполнения
                      запросов к БД.
    """
    async with async_session_maker as session:
        yield session
