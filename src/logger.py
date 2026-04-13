"""
Модуль src.logger.py — настройка логирования для приложения DocAgent‑mini.

Конфигурирует систему логирования на базе Loguru с учётом настроек приложения
(режим отладки, директория и имя файла логов). Обеспечивает централизованную
запись логов в файл с соответствующим уровнем детализации.
"""

from loguru import logger

from src.settings import Settings, get_settings


settings: Settings = get_settings()
"""Настройки приложения, загруженные через get_settings(). Используются
для определения параметров логирования (директория, имя файла, уровень)."""

logger.remove()
"""Удаляет все существующие обработчики логов (sinks) по умолчанию,
чтобы избежать дублирования записей и обеспечить чистую конфигурацию."""

logger.add(
    sink=f'{settings.LOG_DIR}/{settings.LOG_FILENAME}',
    level='TRACE' if settings.DEBUG else 'INFO',
    rotation='1 MB',
    retention='7 days'
)
"""Добавляет обработчик логов, направляющий записи в файл.

Параметры:
* sink: путь к файлу логов (формируется из LOG_DIR и LOG_FILENAME);
* level: уровень логирования — TRACE в режиме отладки (DEBUG=True),
  INFO в обычном режиме.
"""
