"""
tests.api.endpoints.test_authentication

Модуль тестов для эндпоинтов аутентификации в проекте DocAgent‑mini.

Проверяет корректность работы маршрутов, связанных с управлением учётными
записями и сессиями пользователей: регистрацию, аутентификацию (логин),
выход из системы (логаут).
"""

import pytest
from fastapi import status


class TestAuthenticationEndpoints:
    @pytest.mark.anyio
    async def test_register_valid_user(self, db_enabled_test_client):
        """Тест регистрации валидного обычного пользователя."""
        response = db_enabled_test_client.post(
            "/auth/jwt/register",
            json={
                "email": "user@test.com",
                "password": "securepassword123",
                "is_active": True,
                "is_superuser": False,
            },
        )
        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["email"] == "user@test.com"
        assert "id" in data

    @pytest.mark.anyio
    async def test_register_invalid_data(self, db_enabled_test_client):
        """Тест регистрации с невалидными данными (недостающие поля)."""
        # Регистрация без пароля
        response = db_enabled_test_client.post(
            "/auth/jwt/register",
            json={"email": "invalid@test.com"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    @pytest.mark.anyio
    async def test_login_valid_credentials(self, db_enabled_test_client):
        """Тест логина с валидными учётными данными."""
        # Сначала регистрируем пользователя
        db_enabled_test_client.post(
            "/auth/jwt/register",
            json={
                "email": "loginuser@test.com",
                "password": "loginpass789",
                "is_active": True,
                "is_superuser": False,
            },
        )

        # Затем пытаемся залогиниться
        response = db_enabled_test_client.post(
            "/auth/jwt/login",
            data={
                "username": "loginuser@test.com",
                "password": "loginpass789",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.anyio
    async def test_login_invalid_credentials(self, db_enabled_test_client):
        """Тест логина с невалидными учётными данными (неверный пароль)."""
        # Регистрируем пользователя
        db_enabled_test_client.post(
            "/auth/jwt/register",
            json={
                "email": "badlogin@test.com",
                "password": "correctpass",
                "is_active": True,
                "is_superuser": False,
            },
        )

        # Пытаемся залогиниться с неверным паролем
        response = db_enabled_test_client.post(
            "/auth/jwt/login",
            data={
                "username": "badlogin@test.com",
                "password": "wrongpassword",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    @pytest.mark.anyio
    async def test_logout(self, db_enabled_test_client):
        """Тест логаута (требуется валидный токен)."""
        # Регистрируем и логинимся, чтобы получить токен
        db_enabled_test_client.post(
            "/auth/jwt/register",
            json={
                "email": "logoutuser@test.com",
                "password": "logoutpass",
            },
        )
        login_response = db_enabled_test_client.post(
            "/auth/jwt/login",
            data={
                "username": "logoutuser@test.com",
                "password": "logoutpass",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert login_response.status_code == status.HTTP_200_OK
        data = login_response.json()
        assert "access_token" in data
        token = data["access_token"]

        # Выполняем логаут с токеном
        response = db_enabled_test_client.post(
            "/auth/jwt/logout",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == status.HTTP_204_NO_CONTENT

    @pytest.mark.anyio
    async def test_logout_without_token(self, db_enabled_test_client):
        """Тест логаута без передачи токена."""
        response = db_enabled_test_client.post("/auth/jwt/logout")
        # FastAPI-users вернёт 401, если токен не передан
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
