"""
Тесты для модуля loader проекта DocAgent‑mini.

Проверяет корректность инициализации компонента загрузки документации
в рамках функционала RAG (Retrieval‑Augmented Generation).
"""

from src import DocumentationFileLoader, Settings


class TestRAGDocumentationFileLoader:
    """
    Набор тестов для проверки работы загрузчика документации в проекте
    DocAgent‑mini (компонент RAG).
    """

    def test_loader_init_success(self, mock_settings: Settings):
        """
        Тест успешной инициализации загрузчика документации.

        Проверяет, что экземпляр DocumentationFileLoader создаётся
        без ошибок при передаче корректных настроек.
        """
        loader = DocumentationFileLoader(mock_settings)

        assert loader is not None, ('Загрузчик документации не создан')
