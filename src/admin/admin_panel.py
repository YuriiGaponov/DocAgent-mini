"""
src.admin.admin_zone

Модуль инициализации административной зоны для проекта DocAgent‑mini.

Предоставляет фабричные функции для создания и настройки экземпляра SQLAdmin,
который интегрируется в приложение FastAPI и открывает доступ к управлению
моделями БД через веб‑интерфейс.
"""

from typing import List
from fastapi import FastAPI
from sqladmin import Admin, ModelView
from sqlalchemy.ext.asyncio import AsyncEngine

from src.db import providerDB
from src.admin.admin_models import model_views


def _admin_factory(
        app: FastAPI,
        engine: AsyncEngine,
        model_views: List[ModelView]
) -> Admin:
    """
    Внутренняя фабрика для создания и настройки экземпляра SQLAdmin.

    Создаёт объект Admin, связывает его с приложением FastAPI и движком БД,
    а затем регистрирует все представления моделей из переданного списка.

    Args:
        app (FastAPI): экземпляр приложения FastAPI, в который встраивается
            админ‑зона.
        engine (AsyncEngine): асинхронный движок SQLAlchemy, обеспечивающий
            доступ к БД.
        model_views (List[ModelView]): список представлений моделей для
            регистрации в админ‑панели (например, UserAdmin и др.).

    Returns:
        Admin: настроенный экземпляр SQLAdmin с зарегистрированными моделями.
    """
    admin = Admin(app, engine)
    for model_view in model_views:
        admin.add_view(model_view)
    return admin


def get_admin(app: FastAPI) -> Admin:
    """
    Публичный интерфейс для получения настроенного экземпляра SQLAdmin.

    Использует движок БД, предоставленный providerDB, и список представлений
    моделей из admin_models, чтобы создать готовую админ‑панель.

    Args:
        app (FastAPI): экземпляр приложения FastAPI.

    Returns:
        Admin: экземпляр SQLAdmin, интегрированный в приложение и готовый
            к обслуживанию запросов админ‑зоны.
    """
    return _admin_factory(app, providerDB.get_async_engine(), model_views)
