"""
Тесты для эндпоинта /ask API проекта DocAgent‑mini.

Проверяет корректность обработки пользовательских запросов
через HTTP‑интерфейс RAG‑системы.
"""

from http import HTTPStatus
from typing import Dict

import pytest

from fastapi.testclient import TestClient


class TestAskEndpoint:
    """
    Набор тестов для проверки работы эндпоинта /ask в проекте DocAgent‑mini.
    """

    @pytest.mark.asyncio
    async def test_post_ask_endpoint_success(
        self, client: TestClient, ask_test_data: Dict[str, int]
    ):
        """
        Тест успешного выполнения POST‑запроса к эндпоинту /ask.

        Проверяет:
        - отправку POST‑запроса с тестовыми данными пользователя;
        - получение HTTP‑статуса 200 OK в ответе;
        - корректность обработки запроса системой без ошибок.
        """
        response = client.post("/ask", json=ask_test_data)

        assert response.status_code == HTTPStatus.OK, (
            f"Ожидался HTTP‑статус 200 OK, но получен {response.status_code}"
        )
