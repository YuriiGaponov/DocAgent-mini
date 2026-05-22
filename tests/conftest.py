"""
tests.conftest

Модуль conftest тестового пакета проекта DocAgent‑mini.

Содержит фикстуры, используемые в тестах приложения.
"""

from pathlib import Path

import pytest
from alembic import config
from fastapi.testclient import TestClient

from main import app


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
