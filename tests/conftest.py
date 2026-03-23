"""
Модуль tests.conftest.py — конфигурация тестового окружения
для DocAgent‑mini.

Содержит фикстуры pytest для организации тестирования
FastAPI‑приложения. Обеспечивает создание тестового клиента,
который используется во всех интеграционных тестах для имитации
HTTP‑запросов к API.
"""

from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from main import app
from src.settings import Settings


@pytest.fixture
def client() -> TestClient:
    """
    Фикстура pytest для создания тестового HTTP‑клиента.

    Создаёт экземпляр TestClient на базе FastAPI‑приложения (app),
    позволяя отправлять HTTP‑запросы к эндпоинтам в тестовом режиме
    без запуска реального сервера.

    Используется во всех интеграционных тестах проекта для:
    * отправки запросов к API;
    * проверки статусов ответов;
    * валидации структуры и содержимого ответов.

    Returns:
        TestClient: Настроенный тестовый клиент для взаимодействия
            с FastAPI‑приложением.
    """
    return TestClient(app)


@pytest.fixture
def mock_settings() -> Settings:
    """
    Фикстура для создания мок‑настроек приложения.

    Возвращает экземпляр настроек с тестовыми значениями,
    подходящими для запуска тестов. Позволяет изолировать тесты
    от реальных настроек окружения.

    Returns:
        Settings: Экземпляр настроек приложения с тестовыми
            значениями параметров.
    """
    settings = Settings()
    settings.DOC_PATH = "tests/fixtures/test_docs"
    settings.ALLOWED_FILENAME_PATTERN = r"^.*\.md$"
    return settings


@pytest.fixture
def temp_docs_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Фикстура для создания временного каталога с тестовыми документами.

    Создаёт временный каталог и наполняет его тестовыми файлами
    разных типов (валидными и невалидными). Используется для
    тестирования логики обработки документов.

    Args:
        tmp_path (Path): Временный путь, предоставляемый pytest
            для создания временных файлов и каталогов.

    Yields:
        Path: Путь к временному каталогу с тестовыми документами.
            После завершения теста каталог автоматически удаляется.
    """
    docs_dir: Path = tmp_path / "docs"
    docs_dir.mkdir()

    # Создаём тестовые файлы
    (docs_dir / "valid_doc.md").write_text("Content of valid document")
    (docs_dir / "another_doc.md").write_text("Another valid document")
    (docs_dir / "invalid_file.exe").write_text("Should be ignored")

    yield docs_dir


@pytest.fixture
def setup_test_docs_for_get_docs_data(tmp_path: Path) -> Path:
    """
    Создаёт тестовую директорию с документами для get_docs_data.

    Формирует временный каталог с тестовыми Markdown‑файлами,
    содержащими структурированный текст. Используется для проверки
    корректности извлечения данных из документов.

    Args:
        tmp_path (Path): Временный путь от pytest для создания
            тестовых файлов.

    Returns:
        Path: Путь к каталогу с тестовыми документами (doc1.md,
            doc2.md).
    """
    docs_dir = tmp_path / "test_docs_get_data"
    docs_dir.mkdir()

    # Документ 1
    (docs_dir / "doc1.md").write_text(
        "Документ 1\n\n"
        "Содержимое первого документа.\n\n"
        "Важные детали документа 1.",
        encoding='utf-8'
    )
    # Документ 2
    (docs_dir / "doc2.md").write_text(
        "Документ 2\n\n"
        "Содержание второго документа.\n\n"
        "Ключевые моменты документа 2.",
        encoding='utf-8'
    )

    return docs_dir


@pytest.fixture
def temp_file(tmp_path: Path) -> Path:
    """
    Фикстура для создания временного тестового файла.

    Генерирует временный Markdown‑файл с типовым содержимым.
    Используется в тестах, требующих наличия файла для обработки.

    Args:
        tmp_path (Path): Временный путь от pytest для размещения
            тестового файла.

    Returns:
        Path: Полный путь к созданному тестовому файлу
            (test_document.md).
    """
    file_path = tmp_path / "test_document.md"
    content = (
        "Заголовок\n\nПервый абзац.\n\n"
        "Второй абзац с важной информацией.\n\n"
        "Заключение."
    )
    file_path.write_text(content, encoding='utf-8')
    return file_path
