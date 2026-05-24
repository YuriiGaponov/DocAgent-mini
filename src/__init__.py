"""
src.__init__

Пакет src проекта DocAgent‑mini.

Содержит основную бизнес-логику и компоненты AI‑агента.

Структура пакета обеспечивает разделение ответственности между модулями,
способствует повторному использованию кода и упрощает тестирование
отдельных компонентов.
"""

from src.db import get_providerDB
from src.settings import get_settings

settings = get_settings()
providerDB = get_providerDB(settings)

__all__ = ['providerDB', 'settings']
