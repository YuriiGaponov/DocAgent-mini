"""
tests.test_main_app

Модуль с тестами для проверки работы основного приложения DocAgent‑mini.
"""

from fastapi.testclient import TestClient
from http import HTTPStatus


class TestMainApp:
    """
    Набор тестов для проверки базового функционала приложения.
    """

    def test_app_exists(self, client: TestClient):
        """
        Проверяет доступность эндпоинта /docs.

        Отправляет GET-запрос к эндпоинту /docs и убеждается,
        что сервер возвращает статус OK (200).

        Args:
            client (TestClient): экземпляр клиента для тестирования API.

        Assertion:
            status_code == HTTPStatus.OK
        """
        response = client.get("/docs")
        assert response.status_code == HTTPStatus.OK, (
            'Главное приложение FastAPI недоступно'
        )
