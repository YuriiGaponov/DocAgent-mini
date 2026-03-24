"""
Модуль tests.rag.test_rag_integration.py — интеграционные тесты
для системы RAG (Retrieval‑Augmented Generation) в DocAgent‑mini.

Проверяет сквозную работу RAG‑системы: загрузку документов, извлечение
метаданных, разбиение на чанки и возврат корректных данных.
"""


from datetime import datetime
from pathlib import Path
from typing import List


import pytest

from src.models import DocumentData, DocumentMetadata
from src.rag import RAGSystem
from src.settings import Settings


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
