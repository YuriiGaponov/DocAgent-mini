"""
Модуль conftest для pytest в проекте DocAgent‑mini: содержит общие фикстуры
для тестов.
"""

from datetime import datetime
from pathlib import Path
from typing import Generator, List

import pytest
from fastapi.testclient import TestClient

from main import app
from src import DocumentMetadata, EmbeddedDocument, Settings


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
    по умолчанию для использования в тестах. Устанавливает:
    - путь к документации (DOC_PATH) равным temp_docs_dir;
    - имя векторной БД (VECTOR_DB_NAME) как 'test_docs'.
    """
    settings = Settings()
    settings.DOC_PATH = temp_docs_dir
    settings.VECTOR_DB_NAME = 'test_docs'
    return settings


@pytest.fixture
def one_paragraph_text() -> str:
    """
    Фикстура, предоставляющая тестовый текст из одного абзаца.

    Используется для проверки функциональности обработки текста,
    в т. ч. генерации эмбеддингов.
    """
    return 'Текст, состоящий из одного абзаца.'


@pytest.fixture
def two_paragraph_text() -> str:
    """
    Фикстура, предоставляющая тестовый текст из двух абзацев.

    Используется для проверки обработки многоабзацного текста,
    в т. ч. при генерации эмбеддингов для списков текстов.
    """
    return 'Первый абзац текста.\n\nВторой абзац текста.'


@pytest.fixture
def mock_embedded_docs(temp_docs_dir: Path) -> List[EmbeddedDocument]:
    """
    Фикстура, создающая список тестовых эмбеддированных документов.

    Формирует два объекта EmbeddedDocument с тестовыми данными:
    - метаданные файла (имя, тип, путь, время создания и изменения, размер);
    - фрагменты текста (chunks);
    - хеш‑идентификаторы (hash_ids);
    - векторные представления текста (text_embeddings).

    Используется для тестирования функционала работы с векторной базой данных
    и поиска релевантных фрагментов.
    """
    doc1 = EmbeddedDocument(
        file_metadata=DocumentMetadata(
            name='doc1',
            type='.md',
            path=Path(temp_docs_dir / 'doc1'),
            creation_time=datetime.now(),
            modification_time=datetime.now(),
            size=42
        ).to_dict(),
        chunks=['chunk1', 'chunk2'],
        hash_ids=['412cb322137d81a5', '9118bfec488b6ef5'],
        text_embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    doc2 = EmbeddedDocument(
        file_metadata=DocumentMetadata(
            name='doc2',
            type='.md',
            path=Path(temp_docs_dir / 'doc2'),
            creation_time=datetime.now(),
            modification_time=datetime.now(),
            size=42
        ).to_dict(),
        chunks=['chunk3', 'chunk4'],
        hash_ids=['71a7adca1d7f1f4f', 'a8da422011656d8a'],
        text_embeddings=[[0.7, 0.8, 0.9], [0.11, 0.12, 0.13]]
    )
    return [doc1, doc2]


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
