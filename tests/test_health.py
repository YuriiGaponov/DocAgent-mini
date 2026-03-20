"""
Модуль tests.test_health.py — тесты проверки работоспособности
приложения DocAgent‑mini.

Содержит интеграционные тесты для проверки базовых индикаторов
здоровья сервиса:
* доступность эндпоинта /health;
* корректность ответа о состоянии системы;
* доступность интерактивной документации API (/docs).
"""

from http import HTTPStatus
from fastapi.testclient import TestClient


class TestHealth:
    """Набор тестов для проверки индикаторов здоровья приложения."""

    def test_app_health(self, client: TestClient):
        """
        Тест доступности эндпоинта проверки здоровья.

        Проверяет, что приложение запускается и возвращает статус
        «здорово» через эндпоинт /health.

        Steps:
            1. Отправляет GET‑запрос к /health.
            2. Проверяет статус ответа (200 OK).
            3. Проверяет содержимое JSON‑ответа.

        Expected result:
            * статус ответа — HTTPStatus.OK (200);
            * тело ответа содержит ключ "status" со значением "healthy".

        Args:
            client (TestClient): Тестовый HTTP‑клиент для взаимодействия
                с FastAPI‑приложением (предоставляется фикстурой
                из conftest.py).
        """
        response = client.get("/health")

        assert response.status_code == HTTPStatus.OK, (
            f"Ожидаемый статус ответа эндпоинта /health - {HTTPStatus.OK}, "
            f"фактический статус - {response.status_code}"
        )

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy", (
            f"'status': {data["status"]}"
        )

    def test_docs_available(self, client: TestClient):
        """
        Тест доступности интерактивной документации API.

        Проверяет, что Swagger UI (/docs) доступен и возвращает
        успешный HTTP‑статус.

        Steps:
            1. Отправляет GET‑запрос к /docs.
            2. Проверяет статус ответа (200 OK).

        Expected result:
            * статус ответа — HTTPStatus.OK (200).

        Args:
            client (TestClient): Тестовый HTTP‑клиент для взаимодействия
                с FastAPI‑приложением (предоставляется фикстурой
                из conftest.py).
        """
        response = client.get("/docs")
        assert response.status_code == HTTPStatus.OK, (
            f"Ожидаемый статус ответа эндпоинта /docs - {HTTPStatus.OK}, "
            f"фактический статус - {response.status_code}"
        )
