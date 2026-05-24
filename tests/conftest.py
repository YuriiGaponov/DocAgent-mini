"""
tests.conftest

Модуль conftest тестового пакета проекта DocAgent‑mini.

Содержит фикстуры, используемые в тестах приложения.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from alembic import config
from fastapi.testclient import TestClient

from main import app
from src.settings import Settings


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
    with patch("src.settings.get_settings") as mock_get_settings:
        mock_settings = Settings(
            TITLE="Test DocAgent‑mini",
            DBMS="sqlite",
            DB_DIR=Path(""),
            DB_NAME=":memory",
        )
        mock_get_settings.return_value = mock_settings
        yield mock_settings
