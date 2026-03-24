"""
Модуль tests.rag.test_rag_system.py — тесты для системы RAG
(Retrieval‑Augmented Generation) приложения DocAgent‑mini.

Содержит модульные и интеграционные тесты для проверки:
* логики загрузки и фильтрации документации (DocumentationFileLoader);
* инициализации и работы RAG‑системы (RAGSystem);
* обработки крайних случаев (пустые директории, отсутствующие пути).
"""

from pathlib import Path
from typing import List

import pytest

from src.models import DocumentData, DocumentMetadata
from src.rag import DocumentationFileLoader, RAGSystem
from src.settings import Settings


class TestRAGSystem:
    """
    Набор тестов для класса RAGSystem.

    Проверяет корректность инициализации системы и её взаимодействие
    с загрузчиком документации (DocumentationFileLoader), включая:
    * создание экземпляра загрузчика;
    * загрузку документов через RAG‑систему;
    * обработку ошибок (например, отсутствующей директории).
    """

    @pytest.mark.asyncio
    async def test_rag_system_initializes_with_loader(
        self,
        mock_settings: Settings
    ) -> None:
        """
        Тест инициализации RAG‑системы.

        Проверяет, что система корректно создаёт экземпляр
        загрузчика файлов.
        """
        rag = RAGSystem(mock_settings)
        assert rag.fileloader is not None
        assert isinstance(rag.fileloader, DocumentationFileLoader)

    @pytest.mark.asyncio
    async def test_rag_get_docs_calls_loader_with_real_loader(
        self,
        mock_settings: Settings,
        temp_docs_dir: Path
    ) -> None:
        """
        Тест получения документов через RAG‑систему.

        Проверяет интеграцию RAGSystem с реальным загрузчиком
        файлов.
        """
        # Используем реальный экземпляр loader вместо мока
        mock_settings.DOC_PATH = str(temp_docs_dir)
        rag = RAGSystem(mock_settings)

        result: List[Path] = await rag.get_docs()

        # Проверяем результат
        assert len(result) >= 2
        filenames: List[str] = [path.name for path in result]
        assert "valid_doc.md" in filenames
        assert "another_doc.md" in filenames

    @pytest.mark.asyncio
    async def test_rag_get_docs_handles_missing_directory(
        self,
        mock_settings: Settings
    ) -> None:
        """
        Тест обработки отсутствующей директории.

        Проверяет, что система выбрасывает FileNotFoundError
        при попытке загрузить документы из несуществующей папки.
        """
        # Симулируем отсутствие папки с документами
        mock_settings.DOC_PATH = "/nonexistent/path"
        rag = RAGSystem(mock_settings)

        with pytest.raises(FileNotFoundError):
            await rag.get_docs()

    @pytest.mark.asyncio
    async def test_get_docs_data_returns_document_data_list(
        self,
        mock_settings: Settings,
        setup_test_docs_for_get_docs_data: Path
    ) -> None:
        """
        Тест: get_docs_data возвращает список DocumentData.

        Проверяет, что метод возвращает корректный список объектов
        DocumentData при наличии валидных документов.
        """
        mock_settings.DOC_PATH = str(setup_test_docs_for_get_docs_data)
        rag = RAGSystem(mock_settings)

        docs_data = await rag.get_docs_data()

        assert isinstance(docs_data, list)
        assert len(docs_data) == 2  # Два файла в директории
        assert all(isinstance(doc, DocumentData) for doc in docs_data)

    @pytest.mark.asyncio
    async def test_get_docs_data_contains_correct_metadata(
        self,
        mock_settings: Settings,
        setup_test_docs_for_get_docs_data: Path
    ) -> None:
        """
        Тест: метаданные в DocumentData корректны.

        Проверяет заполнение и корректность метаданных для
        каждого документа.
        """
        mock_settings.DOC_PATH = str(setup_test_docs_for_get_docs_data)
        rag = RAGSystem(mock_settings)

        docs_data = await rag.get_docs_data()
        doc_names = [doc.file_metadata.name for doc in docs_data]

        assert "doc1.md" in doc_names
        assert "doc2.md" in doc_names

        for doc in docs_data:
            assert isinstance(doc.file_metadata, DocumentMetadata)
            assert doc.file_metadata.size > 0
            assert doc.file_metadata.path.exists()
            assert doc.file_metadata.type == ".md"

    @pytest.mark.asyncio
    async def test_get_docs_data_chunks_are_correct(
        self,
        mock_settings: Settings,
        setup_test_docs_for_get_docs_data: Path
    ) -> None:
        """
        Тест: чанки разбиты корректно и содержат ожидаемый контент.

        Проверяет качество разбиения текста на чанки и их содержимое.
        """
        mock_settings.DOC_PATH = str(setup_test_docs_for_get_docs_data)
        rag = RAGSystem(mock_settings)

        docs_data = await rag.get_docs_data()

        for doc in docs_data:
            chunks = doc.chunked_text
            assert isinstance(chunks, list)
            assert len(chunks) >= 2  # Минимум 2 абзаца в каждом документе

            # Проверяем, что хотя бы один чанк содержит ключевую фразу
            has_key_phrase = any(
                "документ" in chunk.lower() for chunk in chunks
            )
            assert has_key_phrase is True
            # Проверяем, что чанки не пустые
            assert all(len(chunk.strip()) > 0 for chunk in chunks)

    @pytest.mark.asyncio
    async def test_get_docs_data_handles_empty_directory(
        self,
        mock_settings: Settings,
        tmp_path: Path
    ) -> None:
        """
        Тест обработки пустой директории для get_docs_data.

        Проверяет, что при отсутствии документов возвращается
        пустой список.
        """
        empty_dir = tmp_path / "empty_docs"
        empty_dir.mkdir()
        mock_settings.DOC_PATH = str(empty_dir)
        rag = RAGSystem(mock_settings)

        docs_data = await rag.get_docs_data()

        assert docs_data == []  # Пустой список для пустой папки

    @pytest.mark.asyncio
    async def test_get_docs_data_handles_no_valid_files(
        self,
        mock_settings: Settings,
        tmp_path: Path
    ) -> None:
        """
        Тест когда в директории нет валидных файлов
        (только .exe, .txt и т. д.).

        Проверяет, что система корректно обрабатывает случай, когда
        в директории присутствуют только файлы с запрещёнными
        расширениями.
        """
        invalid_dir = tmp_path / "invalid_docs"
        invalid_dir.mkdir()
        # Создаём только невалидные файлы
        (invalid_dir / "script.py").write_text("Python code")
        (invalid_dir / "data.json").write_text('{"key": "value"}')

        mock_settings.DOC_PATH = str(invalid_dir)
        rag = RAGSystem(mock_settings)

        docs_data = await rag.get_docs_data()

        assert docs_data == []  # Должен вернуть пустой список
