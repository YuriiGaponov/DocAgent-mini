"""
Модуль src.rag.reader.py — читатель файлов документации для системы RAG
(Retrieval‑Augmented Generation) в DocAgent‑mini.

Содержит класс DocumentationFileReader для извлечения данных из файлов:
* метаданных (имя, расширение, время создания/изменения, размер);
* текстового содержимого (UTF‑8).

Формирует структурированные объекты ReadedDocument (текст + метаданные)
для пайплайна RAG: подготовки данных перед поиском фрагментов
и генерацией ответов.
"""

from datetime import datetime
from pathlib import Path

from src.logger import logger
from src.models import DocumentMetadata, ReadedDocument


class DocumentationFileReader:
    """
    Читатель файлов документации. Извлекает метаданные и текстовое содержимое,
    формирует объекты ReadedDocument.

    Предоставляет методы для:
    * получения метаданных файла (имя, тип, время создания/изменения, размер);
    * асинхронного чтения текстового содержимого файла (UTF‑8);
    * формирования структурированного объекта ReadedDocument,
      объединяющего текст и метаданные.
    """

    def get_file_metadata(self, file_path: Path) -> DocumentMetadata:
        """
        Извлекает метаданные файла по указанному пути.
        Создаёт объект DocumentMetadata на основе информации о файле.
        """
        logger.debug('Запуск DocumentationFileReader.get_file_metadata')
        name = file_path.name
        type = file_path.suffix
        stats = file_path.stat()
        creation_time = datetime.fromtimestamp(stats.st_birthtime)
        modification_time = datetime.fromtimestamp(stats.st_mtime)
        size = stats.st_size
        logger.debug(f'Получены мета-данные {file_path.name}')
        try:
            doc_metadata = DocumentMetadata(
                name=name,
                type=type,
                path=file_path,
                creation_time=creation_time,
                modification_time=modification_time,
                size=size
            )
            logger.debug(f'Создан объект {doc_metadata.__class__}')
            return doc_metadata
        except Exception as e:
            logger.error(f'объект класса DocumentMetadata не создан {e}')
            raise

    async def read_text(self, doc_path: Path) -> str:
        """
        Асинхронно читает содержимое текстового файла в кодировке UTF‑8.
        """
        logger.debug('Запуск DocumentationFileReader.read_text')
        with open(doc_path, 'r', encoding='utf-8') as file:
            logger.debug(f'Чтение содержания {file.name}')
            return file.read()

    async def read_file(self, file_path: Path) -> ReadedDocument:
        """
        Читает файл и формирует объект ReadedDocument, объединяющий текст
        и метаданные документа.

        Выполняет последовательное чтение текста и получение метаданных, затем
        создаёт структурированный объект.
        """
        logger.debug('Запуск DocumentationFileReader.read_file')
        text = await self.read_text(file_path)
        meta = self.get_file_metadata(file_path)
        try:
            readed_doc = ReadedDocument(
                file_text=text,
                file_metadata=meta
            )
            return readed_doc
        except Exception as e:
            logger.error(f'объект класса ReadedDocument не создан {e}')
            raise
