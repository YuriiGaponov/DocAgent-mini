"""
Корневой пакет проекта DocAgent‑mini.

Импортирует настройки из модуля main и устанавливает версию пакета,
заимствуя значение из параметра VERSION объекта settings.
"""


from main import settings

__version__ = settings.VERSION
