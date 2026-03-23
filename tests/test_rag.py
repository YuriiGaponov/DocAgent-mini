"""
Модуль tests.test_rag.py — тесты для системы RAG
(Retrieval‑Augmented Generation) приложения DocAgent‑mini.

Содержит модульные и интеграционные тесты для проверки:
* логики загрузки и фильтрации документации (DocumentationFileLoader);
* инициализации и работы RAG‑системы (RAGSystem);
* обработки крайних случаев (пустые директории, отсутствующие пути).
"""

from datetime import datetime
from pathlib import Path
from typing import List

import pytest

from src.models import DocumentData, DocumentMetadata
from src.rag import DocumentationFileLoader, DocumentationFileReader, RAGSystem
from src.settings import Settings


class TestDocumentationFileLoader:
    """
    Набор тестов для класса DocumentationFileLoader.

    Проверяет корректность работы загрузчика документации:
    * фильтрацию файлов по расширениям;
    * проверку безопасности путей;
    * формирование путей к файлам;
    * загрузку документов из директории с учётом фильтров.
    """

    def test_is_filename_allowed_valid(self, mock_settings: Settings) -> None:
        """
        Тест проверки допустимости имени файла (валидные случаи).

        Проверяет, что файлы с разрешённым расширением (.md) проходят фильтр.
        """
        loader = DocumentationFileLoader(mock_settings)
        assert loader.is_filename_allowed("document.md") is True
        assert loader.is_filename_allowed("notes.md") is True

    def test_is_filename_allowed_invalid(
        self, mock_settings: Settings
    ) -> None:
        """
        Тест проверки допустимости имени файла (невалидные случаи).

        Проверяет, что файлы с запрещёнными расширениями отфильтровываются.
        """
        loader = DocumentationFileLoader(mock_settings)
        assert loader.is_filename_allowed("script.py") is False
        assert loader.is_filename_allowed("data.json") is False

    def test_get_file_path_returns_path(self, mock_settings: Settings) -> None:
        """
        Тест создания пути к файлу.

        Проверяет, что метод get_file_path корректно формирует
        объект Path и сохраняет имя файла.
        """
        loader = DocumentationFileLoader(mock_settings)
        result: Path = loader.get_file_path("test.md")
        assert isinstance(result, Path)
        assert result.name == "test.md"

    def test_is_filepath_safe_within_base_dir(
        self,
        mock_settings: Settings,
        temp_docs_dir: Path
    ) -> None:
        """
        Тест безопасности пути внутри базовой директории.

        Проверяет, что путь внутри DOC_PATH считается безопасным.
        """
        mock_settings.DOC_PATH = str(temp_docs_dir)
        loader = DocumentationFileLoader(mock_settings)

        safe_path: Path = temp_docs_dir / "document.md"
        assert loader.is_filepath_safe(safe_path) is True

    def test_is_filepath_safe_outside_base_dir(
        self,
        mock_settings: Settings,
        temp_docs_dir: Path
    ) -> None:
        """
        Тест безопасности пути вне базовой директории.

        Проверяет, что внешние пути блокируются системой безопасности.
        """
        mock_settings.DOC_PATH = str(temp_docs_dir)
        loader = DocumentationFileLoader(mock_settings)

        unsafe_path: Path = Path("/etc/passwd")  # Пример внешнего пути
        assert loader.is_filepath_safe(unsafe_path) is False

    @pytest.mark.asyncio
    async def test_get_docs_filters_files_correctly(
        self,
        mock_settings: Settings,
        temp_docs_dir: Path
    ) -> None:
        """
        Тест фильтрации файлов при загрузке документации.

        Проверяет, что:
        * загружаются только файлы с разрешёнными расширениями (.md);
        * все возвращённые пути безопасны (внутри DOC_PATH).
        """
        mock_settings.DOC_PATH = str(temp_docs_dir)
        loader = DocumentationFileLoader(mock_settings)

        result: List[Path] = await loader.get_docs()

        # Проверяем, что вернулись только .md файлы и они безопасны
        assert len(result) == 2  # Два .md файла
        for path in result:
            assert path.suffix == ".md"
            assert temp_docs_dir in path.parents  # Путь внутри base_dir

    @pytest.mark.asyncio
    async def test_get_docs_empty_directory(
        self,
        mock_settings: Settings,
        tmp_path: Path
    ) -> None:
        """
        Тест загрузки из пустой директории.

        Проверяет, что при отсутствии файлов возвращается пустой список.
        """
        empty_dir: Path = tmp_path / "empty_docs"
        empty_dir.mkdir()
        mock_settings.DOC_PATH = str(empty_dir)

        loader = DocumentationFileLoader(mock_settings)
        result: List[Path] = await loader.get_docs()
        assert result == []  # Пустой список для пустой папки


class TestDocumentationFileReader:
    """
    Набор тестов для класса DocumentationFileReader.
    Проверяет корректность чтения файлов, извлечения метаданных
    и разбиения на чанки.
    """

    def test_get_file_metadata(self, temp_file: Path):
        """Тест извлечения метаданных файла."""
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
        """Тест чтения содержимого файла."""
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
        """Тест обработки отсутствующего файла."""
        reader = DocumentationFileReader()
        nonexistent_path = tmp_path / "nonexistent.md"

        with pytest.raises(FileNotFoundError):
            await reader.read_file(nonexistent_path)

    @pytest.mark.asyncio
    async def test_get_chunks(self, temp_file: Path):
        """Тест разбиения текста на чанки (по абзацам)."""
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
        """Тест разбиения пустого файла на чанки."""
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("", encoding='utf-8')

        reader = DocumentationFileReader()
        chunks = await reader.get_chunks(empty_file)

        assert chunks == ['']

    @pytest.mark.asyncio
    async def test_get_file_data(self, temp_file: Path):
        """
        Полный тест метода get_file_data — получение всех данных документа.
        """
        reader = DocumentationFileReader()
        file_data = await reader.get_file_data(temp_file)

        assert isinstance(file_data, DocumentData)
        assert isinstance(file_data.file_metadata, DocumentMetadata)
        assert file_data.file_metadata.name == "test_document.md"
        assert "Заголовок" in file_data.file_text
        assert len(file_data.chunked_text) == 4
        assert all(isinstance(chunk, str) for chunk in file_data.chunked_text)


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

        Проверяет, что система корректно создаёт экземпляр загрузчика файлов.
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

        Проверяет интеграцию RAGSystem с реальным загрузчиком файлов.
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
    ):
        """Тест: get_docs_data возвращает список DocumentData."""
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
    ):
        """Тест: метаданные в DocumentData корректны."""
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
    async def test_get_docs_data_contains_full_text(
        self,
        mock_settings: Settings,
        setup_test_docs_for_get_docs_data: Path
    ):
        """Тест: полный текст документа корректно извлечён."""
        mock_settings.DOC_PATH = str(setup_test_docs_for_get_docs_data)
        rag = RAGSystem(mock_settings)

        docs_data = await rag.get_docs_data()

        for doc in docs_data:
            full_text = doc.file_text
            assert isinstance(full_text, str)
            assert len(full_text) > 10  # Текст не пустой
            # Проверяем, что текст содержит заголовок (первая строка)
            first_line = full_text.split('\n')[0]
            assert "Документ" in first_line

    @pytest.mark.asyncio
    async def test_get_docs_data_chunks_are_correct(
        self,
        mock_settings: Settings,
        setup_test_docs_for_get_docs_data: Path
    ):
        """Тест: чанки разбиты корректно и содержат ожидаемый контент."""
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
    ):
        """Тест обработки пустой директории для get_docs_data."""
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
    ):
        """
        Тест когда в директории нет валидных файлов
        (только .exe, .txt и т. д.).
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


@pytest.mark.asyncio
async def test_full_rag_flow_with_real_files(
    mock_settings: Settings,
    temp_docs_dir: Path
) -> None:
    """
    Тест полного потока работы RAG с реальными файлами.

    Проверяет сквозную работу системы: загрузку, фильтрацию
    и возврат корректного набора документов, включая извлечение
    метаданных и разбиение на чанки.
    """
    mock_settings.DOC_PATH = str(temp_docs_dir)
    rag = RAGSystem(mock_settings)

    # Шаг 1: Получаем список безопасных путей к документам
    doc_paths: List[Path] = await rag.get_docs()

    # Проверяем результат загрузки путей
    assert len(doc_paths) == 2
    filenames: List[str] = [path.name for path in doc_paths]
    assert "valid_doc.md" in filenames
    assert "another_doc.md" in filenames
    assert "invalid_file.exe" not in filenames  # Отфильтрован

    # Шаг 2: Получаем полные данные документов (метаданные + текст + чанки)
    docs_data: List[DocumentData] = await rag.get_docs_data()

    # Проверяем, что получили данные для всех валидных документов
    assert isinstance(docs_data, list)
    assert len(docs_data) == 2
    assert all(isinstance(doc, DocumentData) for doc in docs_data)

    # Собираем имена документов из метаданных для проверки
    metadata_filenames: List[str] = [
        doc.file_metadata.name for doc in docs_data
    ]
    assert "valid_doc.md" in metadata_filenames
    assert "another_doc.md" in metadata_filenames

    # Шаг 3: Детальная проверка данных каждого документа
    for doc in docs_data:
        # Проверяем метаданные
        metadata = doc.file_metadata
        assert isinstance(metadata, DocumentMetadata)
        assert metadata.name in ["valid_doc.md", "another_doc.md"]
        assert metadata.type == ".md"
        assert metadata.size > 0
        assert metadata.path.exists()
        assert isinstance(metadata.creation_time, datetime)
        assert isinstance(metadata.modification_time, datetime)

        # Проверяем полный текст
        full_text = doc.file_text
        assert isinstance(full_text, str)
        assert len(full_text) > 10  # Текст не пустой

        # Проверка первой строки
        first_line = full_text.split('\n')[0].strip()
        assert len(first_line) > 0  # Первая строка не пустая
        assert any(c.isalpha() for c in first_line)  # Содержит буквы

        # Проверяем чанки
        chunks = doc.chunked_text
        assert isinstance(chunks, list)
        assert len(chunks) >= 1  # Минимум 1 чанк (если файл не разбивается)

        # Проверяем, что хотя бы один чанк содержит значимый текст
        has_meaningful_content = any(
            len(chunk.strip()) > 5 and any(c.isalpha() for c in chunk)
            for chunk in chunks
        )
        assert has_meaningful_content is True

        # Проверяем, что все чанки не пустые после обрезки пробелов
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

        # Проверяем, что все чанки — строки
        assert all(isinstance(chunk, str) for chunk in chunks)

    # Шаг 4: Проверяем интеграцию компонентов
    # Убеждаемся, что пути из get_docs совпадают
    # с путями в метаданных get_docs_data
    path_set_from_get_docs = set(doc_paths)
    path_set_from_metadata = set(
        doc.file_metadata.path for doc in docs_data
    )

    assert path_set_from_get_docs == path_set_from_metadata

    # Дополнительно проверяем, что все пути безопасны
    for doc_path in doc_paths:
        assert rag.fileloader.is_filepath_safe(doc_path) is True
