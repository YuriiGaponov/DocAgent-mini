"""
src.admin.admin_models

Модуль моделей административной панели для проекта DocAgent‑mini.

Определяет классы представлений (ModelView) для SQLAdmin, которые описывают,
как сущности БД будут отображаться и редактироваться в админ‑интерфейсе.
"""

from typing import List

from sqladmin import ModelView

from src.db import User


class UserAdmin(ModelView, model=User):
    """
    Представление модели User в административной панели.

    Определяет, какие поля модели User будут отображаться в списке записей
    в интерфейсе SQLAdmin.

    Attributes:
        column_list (list): перечень полей, отображаемых в таблице списка
                            пользователей.
    """
    column_list = [User.id, User.email]


"""Список представлений моделей для регистрации в SQLAdmin."""
model_views: List[ModelView] = [UserAdmin]
