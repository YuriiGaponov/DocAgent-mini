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
