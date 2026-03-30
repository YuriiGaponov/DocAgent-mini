"""
Пакет src — основной пакет приложения DocAgent‑mini.

Содержит все внутренние модули проекта, организованные по функциональному
принципу: настройки, логирование, обработка запросов, работа с документами
и управление задачами.

Файл __init__.py выполняет роль маркера пакета Python для директории src/
и может в дальнейшем использоваться для:
* экспорта ключевых классов и функций на уровень пакета;
* настройки глобальной конфигурации пакета;
* определения публичного API пакета.

В текущей реализации файл остаётся пустым, обеспечивая базовую функцию —
маркировку директории как Python‑пакета для корректной работы импортов.
"""


from src.models import DocumentMetadata, EmbeddedDocument, ReadedDocument
from src.settings import Settings
from src.rag.embedding_manager import EmbeddingService
from src.rag.loader import DocumentationFileLoader
from src.rag.reader import DocumentationFileReader
from src.rag.vectorDB_manager import VectorDBManager

__all__ = [
    'DocumentationFileLoader', 'DocumentationFileReader', 'DocumentMetadata',
    'EmbeddingService', 'EmbeddedDocument', 'ReadedDocument', 'Settings',
    'VectorDBManager'
]
