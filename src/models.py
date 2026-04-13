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
from langchain_core.messages import AnyMessage
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


class AskRequest(BaseModel):
    """
    Модель запроса от пользователя для RAG‑системы проекта DocAgent‑mini.

    Содержит:
    - идентификатор пользователя (user_id) — уникальный номер,
      идентифицирующий пользователя в системе;
    - текстовый запрос (query) — вопрос или инструкция от пользователя,
      которую система должна обработать для поиска релевантной информации
      в документации и генерации ответа.
    """
    user_id: int
    query: str


class State(BaseModel):
    """
    Модель состояния диалога/задачи для агента DocAgent‑mini.

    Описывает текущее состояние взаимодействия, включая:
    - идентификатор пользователя (user_id);
    - опциональный идентификатор задачи (task_id), который может быть
      не задан (None) на начальных этапах;
    - историю сообщений (messages) в виде списка объектов AnyMessage,
      с поддержкой механизма добавления сообщений через add_messages.
    """
    user_id: int
    task_id: int | None = None
    messages: List[AnyMessage] = []
