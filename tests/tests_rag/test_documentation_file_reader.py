"""
Модуль tests.rag.test_documentation_file_reader.py — тесты
для класса DocumentationFileReader.

Проверяет корректность работы модуля чтения документации:
* извлечение метаданных файла;
* чтение содержимого файлов;
* обработку ошибок (например, отсутствующих файлов);
* разбиение текста на чанки по абзацам;
* формирование полной структуры данных документа.
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.models import DocumentData, DocumentMetadata
from src.rag import DocumentationFileReader


class TestDocumentationFileReader:
    """
    Набор тестов для класса DocumentationFileReader.

    Проверяет корректность чтения файлов, извлечения
    метаданных и разбиения на чанки.
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
    async def test_read_file(self, temp_file: Path):
        """
        Тест чтения содержимого файла.

        Проверяет, что метод read_file корректно считывает
        текст из файла и возвращает его без искажений.

        Args:
            temp_file (Path): Временный тестовый файл,
                предоставляемый фикстурой.
        """
        reader = DocumentationFileReader()
        content = await reader.read_file(temp_file)

        expected_content = (
            "Заголовок\n\nПервый абзац.\n\n"
            "Второй абзац с важной информацией.\n\n"
            "Заключение."
        )
        assert content == expected_content

    @pytest.mark.asyncio
    async def test_read_file_nonexistent(self, tmp_path: Path):
        """
        Тест обработки отсутствующего файла.

        Проверяет, что при попытке чтения несуществующего
        файла возникает исключение FileNotFoundError.

        Args:
            tmp_path (Path): Временная директория,
                предоставляемая фикстурой.
        """
        reader = DocumentationFileReader()
        nonexistent_path = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            await reader.read_file(nonexistent_path)

    @pytest.mark.asyncio
    async def test_get_chunks(self, temp_file: Path):
        """
        Тест разбиения текста на чанки (по абзацам).

        Проверяет, что метод get_chunks корректно разбивает
        содержимое файла на отдельные абзацы.

        Args:
            temp_file (Path): Временный тестовый файл,
                предоставляемый фикстурой.
        """
        reader = DocumentationFileReader()
        chunks = await reader.get_chunks(temp_file)

        assert isinstance(chunks, list)
        assert len(chunks) == 4  # 4 абзаца в тестовом файле
        assert "Заголовок" in chunks[0]
        assert "Первый абзац." in chunks[1]
        assert "Второй абзац с важной информацией." in chunks[2]
        assert "Заключение." in chunks[3]

    @pytest.mark.asyncio
    async def test_get_chunks_empty_file(self, tmp_path: Path):
        """
        Тест разбиения пустого файла на чанки.

        Проверяет поведение метода get_chunks при обработке
        пустого файла. Ожидаемый результат — список с одной
        пустой строкой.

        Args:
            tmp_path (Path): Временная директория,
                предоставляемая фикстурой.
        """
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding='utf-8')

        reader = DocumentationFileReader()
        chunks = await reader.get_chunks(empty_file)

        assert chunks == ['']

    @pytest.mark.asyncio
    async def test_get_file_data(self, temp_file: Path):
        """
        Полный тест метода get_file_data — получение всех данных документа.

        Проверяет комплексную работу метода, который объединяет
        метаданные, полный текст и разбиение на чанки в единую
        структуру DocumentData.

        Args:
            temp_file (Path): Временный тестовый файл,
                предоставляемый фикстурой.
        """
        reader = DocumentationFileReader()
        file_data = await reader.get_file_data(temp_file)

        assert isinstance(file_data, DocumentData)
        assert isinstance(file_data.file_metadata, DocumentMetadata)
        assert file_data.file_metadata.name == "test_document.md"
        assert len(file_data.chunked_text) == 4
        assert all(isinstance(chunk, str) for chunk in file_data.chunked_text)
