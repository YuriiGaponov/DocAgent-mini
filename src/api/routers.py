"""src.api.routers"""

from fastapi import APIRouter

from src.api.endpoints import auth_router


"""Главный роутер приложения."""
router = APIRouter()

router.include_router(auth_router)
