"""
Модуль conftest для pytest в проекте DocAgent‑mini: содержит общие фикстуры
для тестов.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from src import Settings


@pytest.fixture
def client() -> TestClient:
    """
    Фикстура для создания тестового клиента FastAPI в проекте DocAgent‑mini.

    Возвращает экземпляр TestClient, инициализированный
    с приложением app из модуля main.
    """
    return TestClient(app)


@pytest.fixture
def mock_settings() -> Settings:
    """
    Фикстура для создания мок‑настроек проекта DocAgent‑mini.

    Создаёт и возвращает экземпляр класса Settings с настройками
    по умолчанию для использования в тестах.
    """
    settings = Settings()
    return settings
