"""
Модуль src.models.py — модели данных для DocAgent‑mini.

Определяет структуры данных (dataclasses) для представления:
* метаданных документа (DocumentMetadata);
* полных данных документа с чанками (DocumentData)
  для использования в RAG‑системе.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class DocumentMetadata:
    """
    Метаданные документа.

    Хранит основную информацию о файле, необходимую для управления
    документацией и работы RAG‑системы.
    """
    name: str
    """Имя файла без пути."""

    type: str
    """Расширение файла (например, 'md', 'txt')."""

    path: Path
    """Полный путь к файлу (объект Path)."""

    creation_time: datetime
    """Дата и время создания файла."""

    modification_time: datetime
    """Дата и время последнего изменения файла."""

    size: int
    """Размер файла в байтах."""


@dataclass
class DocumentData:
    """
    Структура данных документа для RAG‑системы.

    Содержит полный контент документа, разбитый на чанки,
    и метаданные для поиска и анализа. Используется как основной
    формат представления документа в системе.
    """
    file_metadata: DocumentMetadata
    """Метаданные файла (экземпляр DocumentMetadata)."""

    file_text: str
    """Полный текст документа в виде строки."""

    chunked_text: List[str]
    """Список строк — разбитый на логические части текст документа.
    Каждый элемент списка представляет собой один чанк текста,
    удобный для эмбеддингов и поиска."""
