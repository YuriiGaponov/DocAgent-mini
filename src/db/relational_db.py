"""
src.db.relational_db

Модуль работы с реляционной базой данных в проекте DocAgent‑mini.

Предоставляет инструменты для асинхронного взаимодействия с БД
через SQLAlchemy:
- `PreBase` и `Base`: базовые классы для создания моделей БД.
- `ProviderDB`: управление подключением к БД и создание сессий.
- `get_providerDB`: фабрика провайдеров БД.

Используется для хранения структурированных данных
(задач, истории взаимодействий и т. д.)
в рамках AI‑агента.
"""

from collections.abc import AsyncGenerator

from sqlalchemy import Column, Integer
from sqlalchemy.ext.asyncio import (
    AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
)
from sqlalchemy.orm import declared_attr, declarative_base

from src.logger import app_logger as logger
from src.settings import Settings


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


"""
Декларативная база для моделей SQLAlchemy.

Наследует параметры от `PreBase`, включая автоматическое имя таблицы
и поле `id`. Используется как родительский класс для всех моделей БД.
"""
Base = declarative_base(cls=PreBase)


class ProviderDB:
    """
    Класс, управляющий подключением к реляционной БД и созданием сессий.

    Инкапсулирует логику формирования URL БД, создания асинхронного движка
    и предоставления сессий для работы с данными.

    Args:
        settings (Settings): объект настроек приложения, используемый для
                             конфигурирования подключения к БД.
    """
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        # Словарь `available_db_url` можно расширять для поддержки новых СУБД.
        self.available_db_url: dict[str, str] = {
            'sqlite': (
                f'sqlite+aiosqlite:///'
                f'{self.settings.DB_DIR}/{self.settings.DB_NAME}'
            )
        }
        self._async_engine: AsyncEngine | None = None

    def get_db_url(self) -> str:
        """
        Формирует URL подключения к базе данных на основе настроек проекта.

        Returns:
            str: URL для подключения к выбранной СУБД.

        Raises:
            KeyError: если тип СУБД (settings.DBMS) не найден в
                      `available_db_url`.
        """
        try:
            return self.available_db_url[self.settings.DBMS]
        except KeyError as e:
            logger.error(
                'Неподдерживаемая СУБД',
                extra={
                    'DBMS': self.settings.DBMS,
                    'error': str(e)
                }
            )
            raise

    def get_async_engine(self) -> AsyncEngine:
        """
        Создаёт или возвращает существующий асинхронный движок SQLAlchemy.

        Использует ленивую инициализацию: движок создаётся при первом вызове
        метода и переиспользуется в дальнейшем.

        Returns:
            AsyncEngine: асинхронный движок SQLAlchemy.
        """
        if self._async_engine is None:
            self._async_engine = create_async_engine(self.get_db_url())
        return self._async_engine

    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Генератор асинхронных сессий БД.

        Создаёт и предоставляет асинхронную сессию в контексте async with.

        Yields:
            AsyncSession: активная асинхронная сессия SQLAlchemy для выполнения
                        запросов к БД.
        """
        async_session_maker = async_sessionmaker(
            self.get_async_engine(), expire_on_commit=False
        )
        async with async_session_maker() as session:
            yield session


def get_providerDB(settings: Settings) -> ProviderDB:
    """
    Создаёт экземпляр ProviderDB с переданными настройками.

    Args:
        settings (Settings): объект настроек приложения.

    Returns:
        ProviderDB: инициализированный провайдер БД.
    """
    return ProviderDB(settings)


def create_async_session_dependency(provider_db: ProviderDB):
    """
    Создаёт зависимость для FastAPI на основе экземпляра ProviderDB.

    Позволяет интегрировать асинхронные сессии SQLAlchemy в маршрутизацию
    FastAPI через Dependency Injection. Возвращает асинхронный контекстный
    менеджер, который предоставляет сессию БД для каждого запроса.

    Args:
        provider_db (ProviderDB): экземпляр провайдера БД.

    Returns:
        Callable[[], AsyncSession]: функция‑зависимость, которая при вызове
            создаёт и возвращает активную сессию AsyncSession.
    """
    async def get_async_session() -> AsyncSession:
        async_session_maker = async_sessionmaker(
            provider_db.get_async_engine(), expire_on_commit=False
        )
        async with async_session_maker() as session:
            return session
    return get_async_session
