"""
tests.conftest

Модуль conftest тестового пакета проекта DocAgent‑mini.

Содержит фикстуры, используемые в тестах приложения.
"""

from io import BytesIO
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
from alembic import config
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from main import app
from src.db import Base
from src.settings import Settings


"""URL для тестовой БД in-memory."""
DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def async_session():
    """
    Фикстура для асинхронной SQLAlchemy‑сессии в тестовом контексте.

    Создаёт движок и фабрику сессий для in‑memory SQLite,
    автоматически создаёт все таблицы (на основе Base.metadata)
    перед тестом и корректно освобождает ресурсы после него.

    Yields:
        async_sessionmaker: фабрика асинхронных сессий, готовая к использованию
            в тестах (например, для прямого взаимодействия с БД).
    """
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        future=True,
    )

    # Создаём все таблицы (аналог Base.metadata.create_all)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Фабрика сессий
    async_session = async_sessionmaker(
        engine, expire_on_commit=False
    )

    yield async_session

    # Закрываем движок после теста
    await engine.dispose()


@pytest.fixture
async def db_enabled_test_client(async_session):
    """
    Фикстура тестового клиента FastAPI с включённой поддержкой тестовой БД.

    Переопределяет зависимость async_session_dependency в основном приложении
    на тестовую фабрику сессий, что позволяет тестировать эндпоинты,
    взаимодействующие с БД, в изолированном окружении.

    Args:
        async_session (async_sessionmaker): фабрика асинхронных сессий,
            предоставленная фикстурой async_session.

    Returns:
        TestClient: экземпляр тестового клиента, сконфигурированный
            для работы с основным приложением (main.app), но с подменой
            зависимости БД на тестовую in‑memory сессию.
    """
    from main import app
    from src.db import async_session_dependency

    # Функция-зависимость в виде асинхронного генератора
    async def test_async_session_dep():
        async with async_session() as session:
            yield session

    app.dependency_overrides[async_session_dependency] = test_async_session_dep
    client = TestClient(app)
    return client


@pytest.fixture
def client() -> TestClient:
    """
    Фикстура для создания тестового клиента FastAPI в проекте DocAgent‑mini.

    Возвращает экземпляр TestClient, инициализированный
    с приложением app из модуля main.
    """
    return TestClient(app)


@pytest.fixture(scope="session")
def alembic_config():
    """
    Фикстура для загрузки конфигурации Alembic из файла alembic.ini.
    """
    root_dir = Path(__file__).resolve().parent.parent
    alembic_ini = root_dir / "alembic.ini"
    return config.Config(str(alembic_ini))


@pytest.fixture
def mock_settings():
    """Фикстура - мок настроек приложения."""
    with patch("src.settings.settings") as mock_get_settings:
        mock_settings = Settings(
            TITLE="Test DocAgent‑mini",
            DBMS="sqlite",
            DB_DIR=Path(""),
            DB_NAME=":memory",
        )
        mock_get_settings.return_value = mock_settings
        yield mock_settings


@pytest.fixture
async def test_txt_file():
    """
    Фикстура: UploadFile-подобный объект для тестирования эндпоинтов,
    принимающих файлы.
    """
    file_content = b"Content for UploadFile testing."
    filename = "test_document.txt"
    file_like = BytesIO(file_content)
    file_like.seek(0)
    upload_file = UploadFile(
        filename=filename,
        file=file_like,
        content_type="text/plain"
    )
    return upload_file


@pytest.fixture
def mock_file_open():
    """
    Фикстура для мокирования встроенной функции open() при тестировании работы
    с файлами.

    Используется в тестах, где код открывает файлы
    (через with open(...) или напрямую),
    чтобы избежать реальных операций с файловой системой.
    """
    return mock_open()
