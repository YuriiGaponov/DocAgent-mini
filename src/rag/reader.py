"""
Модуль src.rag.reader.py — читатель файлов документации для системы RAG
(Retrieval‑Augmented Generation) в DocAgent‑mini.

Содержит класс DocumentationFileReader для извлечения данных из файлов:
* метаданных (имя, расширение, время создания/изменения, размер);
* текстового содержимого (UTF‑8).

Формирует структурированные объекты ReadedDocument (текст + метаданные)
для пайплайна RAG: подготовки данных перед поиском фрагментов
и генерацией ответов.

Зависимости:
* src.models.DocumentMetadata — структура метаданных файла;
* src.models.ReadedDocument — структура прочитанного документа;
* src.logger.logger — логгер для отслеживания операций.
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
        Извлекает метаданные файла по пути. Логирует начало и завершение
        операции.
        """
        logger.debug('Запуск DocumentationFileReader.get_file_metadata')
        name = file_path.name
        # logger.trace(f'имя файла {name, name.__class__}')
        type = file_path.suffix
        # logger.trace(f'тип файла {type, type.__class__}')
        # logger.trace(f'путь файла {file_path, file_path.__class__}')
        stats = file_path.stat()
        creation_time = datetime.fromtimestamp(stats.st_birthtime)
        # logger.trace(f'время создания файла {creation_time}')
        modification_time = datetime.fromtimestamp(stats.st_mtime)
        logger.trace(f'время изменения файла {modification_time}')
        size = stats.st_size
        # logger.trace(f'размер файла {size}')
        logger.debug(f'Получены мета-данные {file_path.name}')
        # logger.trace('Начато создание объекта класса DocumentMetadata')
        try:
            # doc_metadata = DocumentMetadata(
            #     name, type, file_path, creation_time, modification_time, size
            # )
            doc_metadata = DocumentMetadata(
                name=name,
                type=type,
                path=file_path,
                creation_time=creation_time,
                modification_time=modification_time,
                size=size
            )
            # return DocumentMetadata(
            #     name, type, file_path, creation_time, modification_time, size
            # )
            logger.debug(f'Создан объект {doc_metadata.__class__}')
            return doc_metadata
        except Exception as e:
            logger.error(f'объект класса DocumentMetadata не создан {e}')
            raise

    async def read_text(self, doc_path: Path) -> str:
        """
        Асинхронно читает содержимое текстового файла (UTF‑8). Логирует
        начало и факт чтения.
        """
        logger.debug('Запуск DocumentationFileReader.read_text')
        with open(doc_path, 'r', encoding='utf-8') as file:
            logger.debug(f'Чтение содержания {file.name}')
            return file.read()

    async def read_file(self, file_path: Path) -> ReadedDocument:
        """
        Читает файл и формирует объект ReadedDocument (текст + метаданные).
        Логирует начало и завершение операции.
        """
        logger.debug('Запуск DocumentationFileReader.read_file')
        logger.trace(f'чтение файла {file_path.name}')
        text = await self.read_text(file_path)
        logger.trace(f'текст файла {text}')
        meta = self.get_file_metadata(file_path)
        logger.trace(f'метаданные файла {meta}')
        logger.trace('Начато создание объекта класса ReadedDocument')
        try:
            # readed_doc = ReadedDocument(text, meta)
            readed_doc = ReadedDocument(
                file_text=text,
                file_metadata=meta
            )
            logger.trace(f'создан объект {readed_doc.__class__}')
            # logger.debug(f'Получены данные {file_path.name}')
            return readed_doc
        except Exception as e:
            logger.error(f'объект класса ReadedDocument не создан {e}')
            raise
