"""
Тесты для модуля vectorDB_manager проекта DocAgent‑mini.

Проверяет корректность работы менеджера векторной базы данных
в рамках функционала RAG (Retrieval‑Augmented Generation).
"""


from typing import List

from src import EmbeddedDocument, Settings, VectorDBManager


class TestRAGVectorDBManager:
    """
    Набор тестов для проверки работы менеджера векторной БД в проекте
    DocAgent‑mini (компонент RAG).
    """

    def test_collection_initiate_success(
            self,
            mock_settings: Settings,
            mock_embedded_docs: List[EmbeddedDocument]
    ):
        """
        Тест успешного добавления документов в векторную БД.

        Проверяет:
        - создание экземпляра VectorDBManager;
        - получение или создание коллекции векторных представлений;
        - корректность имени коллекции;
        - количество документов в коллекции (ожидается 4);
        - возможность получения конкретных записей по ID
          (запрос двух документов и проверка длины результата — 2).
        """
        mock_settings.VECTOR_DB_NAME = 'TestRAGVectorDBManager'
        db_manager = VectorDBManager(mock_settings)
        assert db_manager is not None, 'Менеджер векторной БД не создан'

        collection = db_manager.collection
        assert collection is not None, 'Коллекция не создана'
        assert collection.name == mock_settings.VECTOR_DB_NAME

        db_manager.add_docs_to_collection(mock_embedded_docs)
        assert collection.count() == 4

        records = collection.get(ids=['412cb322137d81a5', '71a7adca1d7f1f4f'])
        assert len(records['ids']) == 2
