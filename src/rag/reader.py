"""
Модуль src.rag.reader.py — читатель файлов документации для системы RAG
(Retrieval‑Augmented Generation) в DocAgent‑mini.

Содержит класс DocumentationFileReader, отвечающий за извлечение данных
из файлов документации. Реализует следующие функции:
* сбор метаданных файла (имя, расширение, время создания и изменения, размер);
* асинхронное чтение текстового содержимого с кодировкой UTF‑8;
* разбиение текста на смысловые блоки (чанки) по двойным переносам строк;
* формирование полной структуры данных документа (DocumentData), объединяющей
  метаданные, исходное содержимое и чанки.

Ключевые методы класса:
* get_file_metadata() — получает метаданные файла по его пути;
* read_file() — асинхронно читает содержимое файла;
* get_chunks() — разбивает текст на чанки;
* get_file_data() — объединяет все данные в объект DocumentData.

Используется в пайплайне RAG как компонент предварительной обработки
документов: подготавливает структурированные данные для последующих
этапов — поиска релевантных фрагментов и генерации ответов.

Зависимости:
* src.models.DocumentData — структура для хранения полных данных
  документа (метаданные + содержимое + чанки);
* src.models.DocumentMetadata — структура для хранения метаданных
  файла (имя, тип, время создания/изменения, размер и т. д.).

Пример использования:
    reader = DocumentationFileReader()
    file_data = await reader.get_file_data(Path("docs/example.md"))
"""


from datetime import datetime
from pathlib import Path
from typing import List

from src.models import DocumentData, DocumentMetadata


class DocumentationFileReader:
    """
    Читатель файлов документации для извлечения метаданных и содержимого.

    Предоставляет методы для:
    * получения метаданных файла (имя, тип, время создания/изменения, размер);
    * чтения текстового содержимого файла;
    * разбиения текста на смысловые блоки (чанки);
    * формирования полной структуры DocumentData для каждого файла.
    """

    def get_file_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Извлекает метаданные файла на основе его пути.

        Собирает информацию о названии, расширении, времени создания и
        изменения, размере файла.

        Args:
            file_path (Path): Полный путь к файлу.

        Returns:
            DocumentMetadata: Объект метаданных документа.
        """
        name = file_path.name
        type = file_path.suffix
        stats = file_path.stat()
        creation_time = datetime.fromtimestamp(stats.st_birthtime)
        modification_time = datetime.fromtimestamp(stats.st_mtime)
        size = stats.st_size
        return DocumentMetadata(
            name, type, file_path, creation_time, modification_time, size
        )

    async def read_file(self, doc_path: Path) -> str:
        """
        Асинхронно читает содержимое текстового файла.

        Открывает файл с кодировкой UTF‑8 и возвращает его содержимое
        в виде строки.

        Args:
            doc_path (Path): Путь к текстовому файлу.

        Returns:
            str: Содержимое файла.
        """
        with open(doc_path, 'r', encoding='utf-8') as file:
            return file.read()

    async def get_chunks(self, doc_path: Path) -> List[str]:
        """
        Асинхронно разбивает содержимое файла на смысловые блоки.

        Разделяет текст по двойным переносам строк (\n\n), формируя
        список чанков для последующего использования в RAG.

        Args:
            doc_path (Path): Путь к текстовому файлу.

        Returns:
            List[str]: Список текстовых блоков (чанков).
        """
        text = await self.read_file(doc_path)
        chunks = text.split('\n\n')
        return chunks

    async def get_file_data(self, file_path: Path) -> DocumentData:
        """
        Асинхронно формирует полную структуру данных документа.

        Объединяет метаданные, исходное содержимое и разбиение на чанки
        в единый объект DocumentData.

        Args:
            file_path (Path): Путь к обрабатываемому файлу.

        Returns:
            DocumentData: Полный объект данных документа, готовый к
                использованию в пайплайне RAG.
        """
        return DocumentData(
            self.get_file_metadata(file_path),
            await self.read_file(file_path),
            await self.get_chunks(file_path)
        )
