"""
Модуль conftest для pytest в проекте DocAgent‑mini: содержит общие фикстуры
для тестов.
"""

from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from main import app
from src import Settings


@pytest.fixture
def client() -> TestClient:
    """
    Фикстура для создания тестового клиента FastAPI в проекте DocAgent‑mini.

    Возвращает экземпляр TestClient, инициализированный
    с приложением app из модуля main.
    """
    return TestClient(app)


@pytest.fixture
def mock_settings(temp_docs_dir: Path) -> Settings:
    """
    Фикстура для создания мок‑настроек проекта DocAgent‑mini.

    Создаёт и возвращает экземпляр класса Settings с настройками
    по умолчанию для использования в тестах. Устанавливает путь к документации
    (DOC_PATH) равным temp_docs_dir.
    """
    settings = Settings()
    settings.DOC_PATH = temp_docs_dir
    return settings


@pytest.fixture
def temp_docs_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Фикстура для создания временного каталога с тестовыми документами.

    Создаёт временный каталог 'docs' и наполняет его тестовыми файлами:
    - valid_doc1.md — валидный Markdown‑документ;
    - valid_doc2.md — валидный Markdown‑документ;
    - not_valid_doc.exe — файл недопустимого расширения.

    Возвращает путь к созданному каталогу. После завершения теста каталог
    автоматически удаляется (управление ресурсами через yield).
    """
    docs_dir: Path = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "valid_doc1.md").write_text("Content of 1th valid document")
    (docs_dir / "valid_doc2.md").write_text("Content of 2nd valid document")
    (docs_dir / "not_valid_doc.exe").write_text(
        "Content of not valid document"
    )
    yield docs_dir
