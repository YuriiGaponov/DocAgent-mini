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

from src.db.relational_db import Base, get_providerDB
from src.settings import settings

providerDB = get_providerDB(settings)

__all__ = ['Base', 'providerDB', 'get_providerDB']
