from datetime import datetime
from pathlib import Path

import pytest

from src.models import DocumentMetadata, ReadedDocument
from src.rag import DocumentationFileReader


class TestDocumentationFileReader:
    """
    Набор тестов для класса DocumentationFileReader.

    Проверяет корректность чтения файлов, извлечения
    метаданных и формирования объектов ReadedDocument.
    """

    def test_get_file_metadata(self, temp_file: Path):
        """
        Тест извлечения метаданных файла.

        Проверяет, что метод get_file_metadata корректно
        извлекает и формирует метаданные из файла.

        Args:
            temp_file (Path): Временный тестовый файл,
                предоставляемый фикстурой.
        """
        reader = DocumentationFileReader()
        metadata = reader.get_file_metadata(temp_file)

        assert isinstance(metadata, DocumentMetadata)
        assert metadata.name == "test_document.md"
        assert metadata.type == ".md"
        assert metadata.path == temp_file
        assert isinstance(metadata.creation_time, datetime)
        assert isinstance(metadata.modification_time, datetime)
        assert metadata.size > 0

    @pytest.mark.asyncio
    async def test_read_text(self, temp_file: Path):
        """
        Тест чтения содержимого файла через read_text.

        Проверяет, что метод read_text корректно считывает
        текст из файла и возвращает его без искажений.

        Args:
            temp_file (Path): Временный тестовый файл,
                предоставляемый фикстурой.
        """
        reader = DocumentationFileReader()
        content = await reader.read_text(temp_file)

        expected_content = (
            "Заголовок\n\nПервый абзац.\n\n"
            "Второй абзац с важной информацией.\n\n"
            "Заключение."
        )
        assert content == expected_content

    @pytest.mark.asyncio
    async def test_read_text_nonexistent(self, tmp_path: Path):
        """
        Тест обработки отсутствующего файла в read_text.

        Проверяет, что при попытке чтения несуществующего
        файла возникает исключение FileNotFoundError.

        Args:
            tmp_path (Path): Временная директория,
                предоставляемая фикстурой.
        """
        reader = DocumentationFileReader()
        nonexistent_path = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            await reader.read_text(nonexistent_path)

    @pytest.mark.asyncio
    async def test_read_file(self, temp_file: Path):
        """
        Полный тест метода read_file — получение данных документа.

        Проверяет комплексную работу метода, который читает текст
        и формирует объект ReadedDocument, объединяющий текст и метаданные.

        Args:
            temp_file (Path): Временный тестовый файл,
                предоставляемый фикстурой.
        """
        reader = DocumentationFileReader()
        readed_doc = await reader.read_file(temp_file)

        assert isinstance(readed_doc, ReadedDocument)
        # Проверка текста
        expected_content = (
            "Заголовок\n\nПервый абзац.\n\n"
            "Второй абзац с важной информацией.\n\n"
            "Заключение."
        )
        assert readed_doc.file_text == expected_content
        # Проверка метаданных
        assert isinstance(readed_doc.file_metadata, DocumentMetadata)
        assert readed_doc.file_metadata.name == "test_document.md"
        assert readed_doc.file_metadata.type == ".md"
        assert readed_doc.file_metadata.path == temp_file

    @pytest.mark.asyncio
    async def test_read_file_nonexistent(self, tmp_path: Path):
        """
        Тест обработки отсутствующего файла в read_file.

        Проверяет, что read_file корректно передаёт исключение
        FileNotFoundError при чтении несуществующего файла.

        Args:
            tmp_path (Path): Временная директория,
                предоставляемая фикстурой.
        """
        reader = DocumentationFileReader()
        nonexistent_path = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            await reader.read_file(nonexistent_path)

    @pytest.mark.asyncio
    async def test_read_empty_file(self, tmp_path: Path):
        """
        Тест чтения пустого файла.

        Проверяет поведение read_file при обработке пустого файла.
        Ожидаемый результат — объект ReadedDocument с пустым текстом.

        Args:
            tmp_path (Path): Временная директория,
                предоставляемая фикстурой.
        """
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding='utf-8')

        reader = DocumentationFileReader()
        readed_doc = await reader.read_file(empty_file)

        assert isinstance(readed_doc, ReadedDocument)
        assert readed_doc.file_text == ""
        assert isinstance(readed_doc.file_metadata, DocumentMetadata)
        assert readed_doc.file_metadata.size == 0
