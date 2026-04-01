"""
Модуль src.rag.utils.py — утилитарные функции для RAG‑системы.

Содержит вспомогательные функции для работы с текстом и генерации
идентификаторов в рамках RAG‑системы (Retrieval‑Augmented Generation)
проекта DocAgent‑mini.
"""

import hashlib
from typing import List


def get_chunks(text: str) -> List[str]:
    """
    Разбивает текст на чанки по двойным переносам строк.

    Возвращает список строк — каждый элемент отдельный чанк.
    """
    return text.split('\n\n')


def generate_hash_id(content: str) -> str:
    """
    Генерирует короткий хеш‑идентификатор (16 символов) для содержимого.

    Использует алгоритм SHA‑256.
    """
    hash_object = hashlib.sha256(content.encode())
    return hash_object.hexdigest()[:16]
