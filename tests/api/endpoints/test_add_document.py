"""
tests.api.endpoints.test_add_document

Модуль тестов для эндпоинта добавления документов в проекте DocAgent‑mini.

Проверяет корректность работы маршрута, отвечающего за загрузку и регистрацию
документов в системе: обработку входных данных, сохранение метаданных,
интеграцию с хранилищем и формирование ответа API.
"""

import pytest


class TestAddDocument:
    """Набор тестов для эндпоинта добавления документа."""

    @pytest.mark.anyio
    async def test_add_valid_document(self, db_enabled_test_client):
        """Тест успешной загрузки валидного документа.

        Проверяет:
        - статус 201 Created;
        - наличие ID и метаданных в ответе;
        - корректность сохранения записи в БД.
        """
        # TODO: реализовать тестовый запрос (например, POST /documents/)
        # с payload, содержащим файл и метаданные
        pass

    @pytest.mark.anyio
    async def test_invalid_payload(self, db_enabled_test_client):
        """Тест обработки невалидного payload (недостающие поля).

        Сценарий: POST /documents/ без обязательных полей.
        Ожидаемый результат: 422 Unprocessable Entity.
        """
        # TODO: реализовать запрос с неполными данными
        pass

    @pytest.mark.anyio
    async def test_unsupported_mime_type(self, db_enabled_test_client):
        """Тест отказа при загрузке документа с неподдерживаемым MIME‑типом.

        Ожидаемый статус: 400 Bad Request или 415 Unsupported Media Type.
        """
        # TODO: реализовать запрос с запрещённым Content-Type
        pass

    @pytest.mark.anyio
    async def test_duplicate_id(self, db_enabled_test_client):
        """
        Тест обработки конфликта — попытка загрузить существующий документ.

        Ожидаемый статус: 409 Conflict.
        """
        # TODO: предварительно создать документ,
        # затем повторить загрузку с тем же ID
        pass
