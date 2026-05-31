"""
src.db.__init__

Пакет реализации работы с базами данных в проекте DocAgent‑mini.

Обеспечивает поддержку двух типов хранилищ данных:
1. Векторная база данных:
для хранения и поиска векторных представлений текстовых документов
(в рамках механизма RAG)

2. Реляционная база данных:
для хранения данных пользователей, метаданных внутренней документации.
"""

from src.db.models import User
from src.db.relational_db import (
    Base, get_providerDB, create_async_session_dependency
)
from src.settings import settings

"""Глобальный экземпляр провайдера реляционной БД."""
providerDB = get_providerDB(settings)

"""Зависимость для получения асинхронной сессии БД."""
async_session_dependency = create_async_session_dependency(providerDB)

__all__ = [
    'Base', 'providerDB', 'get_providerDB',
    'async_session_dependency', 'User'
]
