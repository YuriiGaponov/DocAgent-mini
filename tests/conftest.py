"""
tests.conftest

Модуль conftest тестового пакета проекта DocAgent‑mini.

Содержит фикстуры, используемые в тестах приложения.
"""


import pytest
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
