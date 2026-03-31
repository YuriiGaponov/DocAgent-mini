"""
Модуль src.models.py — модели данных для DocAgent‑mini.

Определяет структуры данных (dataclasses) для представления:
* метаданных документа (DocumentMetadata);
* прочитанного документа (ReadedDocument);
* расширенной структуры с эмбеддингами (EmbeddedDocument)
  для использования в RAG‑системе.
"""

from datetime import datetime
from pathlib import Path
from typing import List

from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """
    Модель метаданных документа для проекта DocAgent‑mini.

    Содержит базовую информацию о документе:
    - имя файла;
    - тип/расширение;
    - полный путь;
    - время создания;
    - время последнего изменения;
    - размер в байтах.
    """
    name: str
    type: str
    path: Path
    creation_time: datetime
    modification_time: datetime
    size: int


class ReadedDocument(BaseModel):
    """
    Модель прочитанного документа для проекта DocAgent‑mini.

    Объединяет:
    - текстовое содержимое файла (file_text);
    - метаданные документа (file_metadata) в формате DocumentMetadata.
    """
    file_text: str
    file_metadata: DocumentMetadata


class EmbeddedDocument(BaseModel):
    """
    Модель документа с эмбеддингами для RAG‑системы проекта DocAgent‑mini.

    Расширенная структура, содержащая:
    - метаданные документа (file_metadata) в формате DocumentMetadata;
    - список текстовых чанков (chunks), на которые разбит документ;
    - список хеш‑идентификаторов (hash_ids) для каждого чанка;
    - список эмбеддингов (text_embeddings) — векторных представлений
      для каждого чанка (каждый эмбеддинг — список чисел с плавающей точкой).
    """
    file_metadata: DocumentMetadata
    chunks: List[str]
    hash_ids: List[str]
    text_embeddings: List[List[float]]
