"""
Тесты для эндпоинта /health проекта DocAgent‑mini.

Проверяет работоспособность API: доступность и корректность ответа
сервисного эндпоинта здоровья системы.
"""

from http import HTTPStatus
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """
    Набор тестов для проверки работоспособности эндпоинта /health
    в проекте DocAgent‑mini.
    """

    def test_get_health_endpoint_success(self, client: TestClient):
        """
        Тест успешного ответа от эндпоинта /health.

        Проверяет:
        - HTTP‑статус 200 OK;
        - наличие поля "status" в ответе;
        - значение "healthy" для поля "status";
        - отсутствие лишних полей в ответе (длина ответа равна 1).
        """
        response = client.get("/health")

        assert response.status_code == HTTPStatus.OK, (
            f"Ожидался HTTP‑статус 200 OK, но получен {response.status_code}"
        )

        response_data = response.json()

        assert "status" in response_data, (
            'В ответе отсутствует поле "status"'
        )
        assert response_data["status"] == "healthy", (
            f'Ожидалось значение "healthy" для поля "status", '
            f'но получено {response_data["status"]}'
        )
        assert len(response_data) == 1, (
            "В ответе присутствуют лишние поля, "
            "которых не должно быть при успешном выполнении"
        )
