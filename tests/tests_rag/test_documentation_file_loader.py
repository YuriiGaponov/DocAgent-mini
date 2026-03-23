"""
Модуль tests.rag.test_documentation_file_loader.py — тесты для класса
DocumentationFileLoader приложения DocAgent‑mini.

Содержит модульные тесты для проверки:
* фильтрации файлов по расширениям;
* проверки безопасности путей;
* формирования путей к файлам;
* загрузки документов из директории с учётом фильтров.
"""


from pathlib import Path
from typing import List

import pytest

from src.rag import DocumentationFileLoader
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
