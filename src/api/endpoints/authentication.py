"""src.api.endpoints.authentication"""

from fastapi import APIRouter

from src.users import auth_backend, fastapi_users

"""Роутер аутентификации."""
router = APIRouter()

router.include_router(
    fastapi_users.get_auth_router(auth_backend)
)
