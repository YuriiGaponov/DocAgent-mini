"""
Тесты для модуля rag_system проекта DocAgent‑mini.

Проверяет корректность работы комплексной RAG‑системы
(Retrieval‑Augmented Generation) в проекте.
"""

import pytest

from src import RAGSystem, Settings


class TestRAGSystem:
    """
    Набор тестов для проверки работы RAG‑системы в проекте DocAgent‑mini.
    """

    @pytest.mark.asyncio
    async def test_create_docs_collection_success(
        self, mock_settings: Settings
    ):
        """
        Тест успешного создания коллекции документов в RAG‑системе.

        Проверяет:
        - инициализацию экземпляра RAGSystem;
        - успешное выполнение метода create_docs_collection();
        - статус результата (ожидается 'success');
        - корректность сообщения о создании коллекции:
          * соответствие имени коллекции значению VECTOR_DB_NAME из настроек;
          * количество созданных записей (ожидается 2).
        """
        rag_sys = RAGSystem(mock_settings)
        assert rag_sys is not None, 'RAG‑система не инициализирована'

        result = await rag_sys.initiate_collection()
        assert result['status'] == 'success'
        assert result['message'] == (
            f'Коллекция {mock_settings.VECTOR_DB_NAME} создана, '
            f'создано {2} записей'
        )
