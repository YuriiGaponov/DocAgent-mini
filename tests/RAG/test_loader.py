"""
Тесты для модуля loader проекта DocAgent‑mini.

Проверяет корректность инициализации и работы компонента загрузки документации
в рамках функционала RAG (Retrieval‑Augmented Generation).
"""

from pathlib import Path
from typing import List
import pytest

from src import DocumentationFileLoader, Settings


class TestRAGDocumentationFileLoader:
    """
    Набор тестов для проверки работы загрузчика документации в проекте
    DocAgent‑mini (компонент RAG).
    """

    @pytest.mark.asyncio
    async def test_loader_success(self, mock_settings: Settings):
        """
        Тест успешной инициализации загрузчика документации
        и его основных методов.

        Проверяет:
        - создание экземпляра DocumentationFileLoader;
        - получение списка документов (ожидается 2 файла);
        - проверку безопасности пути (небезопасный путь должен быть отклонён);
        - проверку допустимости имён файлов (скрипты и исполняемые файлы должны
          быть запрещены).
        """
        loader = DocumentationFileLoader(mock_settings)
        assert loader is not None, 'Загрузчик документации не создан'

        result: List[Path] = await loader.get_docs()
        assert len(result) == 2, (
            'Ожидаемое количество допустимых файлов: 2'
        )

        unsafe_path: Path = Path("/etc/passwd")
        assert loader.is_filepath_safe(unsafe_path) is False

        unsafe_filenames = ['script.py', 'file.exe']
        for name in unsafe_filenames:
            assert loader.is_filename_allowed(name) is False, (
                f'Имя {name} недопустимо'
            )
