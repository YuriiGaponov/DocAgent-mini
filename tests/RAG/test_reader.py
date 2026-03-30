"""
Тесты для модуля reader проекта DocAgent‑mini.

Проверяет корректность работы компонента чтения документации
в рамках функционала RAG (Retrieval‑Augmented Generation).
"""

from pathlib import Path
import pytest

from src import DocumentationFileReader


class TestRAGDocumentationFileReader:
    """
    Набор тестов для проверки работы читателя документации в проекте
    DocAgent‑mini (компонент RAG).
    """

    @pytest.mark.asyncio
    async def test_read_file_success(self, temp_docs_dir: Path):
        """
        Тест успешного чтения файла и получения метаданных.

        Проверяет:
        - создание экземпляра DocumentationFileReader;
        - чтение текстового содержимого Markdown‑файла
            (сравнение с ожидаемым текстом);
        - наличие всех обязательных атрибутов в метаданных файла:
          name, type, path, creation_time, modification_time, size;
        - структуру прочитанного файла (наличие полей file_text и
            file_metadata);
        - корректность текста в поле file_text.
        """
        reader = DocumentationFileReader()
        assert reader is not None, 'Читатель файлов документации не создан'

        assert await reader.read_text(
            Path(temp_docs_dir / 'valid_doc2.md')
        ) == 'Content of 2nd valid document'

        meta = reader.get_file_metadata(
            Path(temp_docs_dir / 'valid_doc2.md')
        )
        assert hasattr(meta, 'name')
        assert hasattr(meta, 'type')
        assert hasattr(meta, 'path')
        assert hasattr(meta, 'creation_time')
        assert hasattr(meta, 'modification_time')
        assert hasattr(meta, 'size')

        readed_file = await reader.read_file(
            Path(temp_docs_dir / 'valid_doc1.md')
        )
        assert hasattr(readed_file, 'file_text')
        assert readed_file.file_text == 'Content of 1th valid document'
        assert hasattr(readed_file, 'file_metadata')
