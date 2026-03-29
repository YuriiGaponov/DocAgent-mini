"""
Модуль src.models.py — модели данных для DocAgent‑mini.

Определяет структуры данных (dataclasses) для представления:
* метаданных документа (DocumentMetadata);
* прочитанного документа (ReadedDocument);
* расширенной структуры с эмбеддингами (EmbeddedDocument)
  для использования в RAG‑системе.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class DocumentMetadata:
    """
    Метаданные документа.

    Хранит основную информацию о файле для управления документацией
    и работы RAG‑системы.
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует объект в словарь для сериализации.

        Конвертирует все поля в базовые типы Python, включая:
        * преобразование Path в строку;
        * форматирование datetime в ISO‑строку.
        """
        return {
            'name': self.name,
            'type': self.type,
            'path': str(self.path),
            'creation_time': self.creation_time.isoformat(),
            'modification_time': self.modification_time.isoformat(),
            'size': self.size
        }


@dataclass
class ReadedDocument:
    """
    Прочитанный документ — объединяет текстовое содержимое и метаданные.

    Используется как промежуточная структура при обработке файлов:
    хранит полный текст и связанные с ним метаданные.
    """
    file_text: str
    """Текстовое содержимое файла."""

    file_metadata: DocumentMetadata
    """Метаданные файла (экземпляр DocumentMetadata)."""


@dataclass
class EmbeddedDocument:
    """
    Расширенная структура данных документа с эмбеддингами.

    Объединяет метаданные, разбиение на чанки и векторные представления
    текста. Используется в RAG‑системе для семантического поиска
    и ранжирования.
    """
    file_metadata: DocumentMetadata
    """Метаданные файла (экземпляр DocumentMetadata)."""

    chunks: List[str]
    """Список строк — разбитый на логические части текст документа.
    Каждый элемент списка — один чанк текста, удобный для эмбеддингов
    и поиска."""

    hash_ids: List[str]
    """Список уникальных идентификаторов для каждого чанка.
    Используются как ключи при добавлении данных в векторную БД."""

    text_embeddings: List[List[float]]
    """Список эмбеддингов (векторных представлений) для каждого чанка.
    Каждый эмбеддинг соответствует своему чанку по индексу."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразует объект в словарь для сериализации.

        Конвертирует все поля в базовые типы Python:
        * преобразует Path в строку в составе file_metadata;
        * форматирует datetime в ISO‑строки в составе file_metadata;
        * гарантирует тип float для значений эмбеддингов.

        Возвращает словарь с данными документа, готовый к сериализации
        (например, в JSON).
        """
        return {
            "file_metadata": {
                "name": self.file_metadata.name,
                "type": self.file_metadata.type,
                "path": str(self.file_metadata.path),
                "creation_time": self.file_metadata.creation_time.isoformat(),
                "modification_time": (
                    self.file_metadata.modification_time.isoformat()
                ),
                "size": self.file_metadata.size
            },
            "chunks": self.chunks,
            "hash_ids": self.hash_ids,
            "text_embeddings": [
                [float(x) for x in embedding]
                for embedding in self.text_embeddings
            ]
        }
