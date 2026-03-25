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

    chunked_text: List[str]
    """Список строк — разбитый на логические части текст документа.
    Каждый элемент списка представляет собой один чанк текста,
    удобный для эмбеддингов и поиска."""


@dataclass
class EmbeddedDocument(DocumentData):
    """
    Расширенная структура данных документа с эмбеддингами.

    Наследует все поля DocumentData и добавляет векторные представления
    (эмбеддинги) для каждого чанка текста.

    Используется в RAG‑системе для семантического поиска и ранжирования.
    """
    text_embeddings: List[List[float]]
    """Список эмбеддингов (векторных представлений) для каждого чанка
    в chunked_text. Каждый эмбеддинг соответствует своему чанку по индексу."""

    def to_dict(self) -> dict:  # временно
        return {
            "file_metadata": {
                "name": self.file_metadata.name,
                "type": self.file_metadata.type,
                "path": str(self.file_metadata.path),  # Path -> str
                "creation_time": self.file_metadata.creation_time.isoformat(),
                "modification_time": (
                    self.file_metadata.modification_time.isoformat()
                ),
                "size": self.file_metadata.size
            },
            "chunked_text": self.chunked_text,
            "text_embeddings": [
                [float(x) for x in embedding]  # Гарантируем тип float
                for embedding in self.text_embeddings
            ]
        }
