"""
Тесты для модуля embedding_manager проекта DocAgent‑mini.

Проверяет корректность работы сервиса генерации эмбеддингов
в рамках функционала RAG (Retrieval‑Augmented Generation).
"""

from typing import List
import pytest

from src import EmbeddingService, Settings


class TestRAGEmbeddingService:
    """
    Набор тестов для проверки работы сервиса генерации эмбеддингов в проекте
    DocAgent‑mini (компонент RAG).
    """

    @pytest.mark.asyncio
    async def test_generate_embedding_success(
        self, mock_settings: Settings,
        one_paragraph_text: str, two_paragraph_text: str
    ):
        """
        Тест успешной генерации эмбеддингов для текста.

        Проверяет:
        - создание экземпляра EmbeddingService;
        - генерацию эмбеддинга для одиночного текстового абзаца
          (проверка типа результата);
        - генерацию эмбеддингов для списка из двух текстов
          (проверка типа и длины результата — ожидается 2 эмбеддинга).
        """
        embedder = EmbeddingService(mock_settings)
        assert embedder is not None, 'Менеджер эмбеддингов не создан'

        one_text_embedding = embedder.generate_embedding(one_paragraph_text)
        assert isinstance(one_text_embedding, List)

        two_text_embedding = embedder.generate_embedding(
            [one_paragraph_text, two_paragraph_text]
        )
        assert isinstance(two_text_embedding, List)
        assert len(two_text_embedding) == 2
