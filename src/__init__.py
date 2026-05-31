"""
src.__init__

Пакет src проекта DocAgent‑mini.

Содержит основную бизнес-логику и компоненты AI‑агента.

Структура пакета обеспечивает разделение ответственности между модулями,
способствует повторному использованию кода и упрощает тестирование
отдельных компонентов.
"""

from src.admin import get_admin
from src.api import router
from src.db import providerDB
from src.docs import tags_metadata
from src.settings import settings

__all__ = ['get_admin', 'providerDB', 'router', 'settings', 'tags_metadata']
