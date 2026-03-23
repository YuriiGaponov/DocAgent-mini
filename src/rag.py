"""
Модуль src.rag.py — реализация системы RAG (Retrieval‑Augmented Generation)
для DocAgent‑mini.

Содержит компоненты для загрузки, чтения и обработки документации:
* DocumentationFileLoader — загрузчик файлов с фильтрацией по расширениям
  и проверкой безопасности путей;
* DocumentationFileReader — читатель файлов, извлекающий метаданные и контент;
* RAGSystem — основная система RAG, интегрирующая загрузчик и читатель
  документов для подготовки данных в пайплайне RAG.
"""
# добавить эмбеддинги файлов

import re
from datetime import datetime
from pathlib import Path
from typing import List

from src.models import DocumentData, DocumentMetadata
from src.settings import Settings


class DocumentationFileLoader:
    """
    Загрузчик документации с фильтрацией файлов и проверкой безопасности путей.


    Обеспечивает безопасную загрузку документов из заданной директории,
    применяя фильтры по расширениям и проверяя, что запрашиваемые файлы
    находятся внутри разрешённой директории.


    Attributes:
        settings (Settings): Настройки приложения, содержащие параметры
            загрузки (путь к документам, шаблон разрешённых имён файлов).
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует загрузчик с заданными настройками.

        Args:
            settings (Settings): Конфигурация приложения (путь к документам,
                шаблон разрешённых имён файлов и т. д.).
        """
        self.settings = settings

    def is_filename_allowed(self, filename: str) -> bool:
        """
        Проверяет, соответствует ли имя файла разрешённому шаблону.

        Использует регулярное выражение из настроек (ALLOWED_FILENAME_PATTERN)
        для фильтрации файлов по расширению.

        Args:
            filename (str): Имя файла для проверки.

        Returns:
            bool: True, если имя файла соответствует шаблону, иначе False.
        """
        if re.match(self.settings.ALLOWED_FILENAME_PATTERN, filename):
            return True
        return False

    def get_file_path(self, filename: str) -> Path:
        """
        Формирует полный безопасный путь к файлу на основе имени.

        Разрешает путь относительно базовой директории документов (DOC_PATH),
        гарантируя нормализацию и разрешение символических ссылок.

        Args:
            filename (str): Имя файла.

        Returns:
            Path: Объект Path, представляющий полный путь к файлу.
        """
        base_dir = Path(self.settings.DOC_PATH).resolve()
        target_path = (base_dir / filename).resolve()
        return target_path

    def is_filepath_safe(self, filepath: Path) -> bool:
        """
        Проверяет безопасность пути к файлу.

        Гарантирует, что запрашиваемый файл находится внутри базовой
        директории документов (DOC_PATH), предотвращая доступ к файлам
        вне этой директории.

        Args:
            filepath (Path): Путь к файлу для проверки.

        Returns:
            bool: True, если путь безопасен (находится внутри DOC_PATH),
                иначе False.
        """
        base_dir = Path(self.settings.DOC_PATH).resolve()
        return str(filepath).startswith(str(base_dir))

    async def get_docs(self) -> List[Path]:
        """
        Асинхронно загружает список безопасных файлов документации.

        Перебирает файлы в директории DOC_PATH, фильтрует их по расширению
        и проверяет безопасность пути. Возвращает список объектов Path
        для валидных файлов.

        Returns:
            List[Path]: Список путей к файлам документации,
                соответствующим критериям фильтрации и безопасности.

        Raises:
            FileNotFoundError: Если директория DOC_PATH не существует.
        """
        def _generate_safe_files():
            dir_path = Path(self.settings.DOC_PATH)

            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")

            for obj in dir_path.iterdir():
                if not obj.is_file():
                    continue

                filename = obj.name
                if not self.is_filename_allowed(filename):
                    continue

                file_path = self.get_file_path(filename)
                if file_path and self.is_filepath_safe(file_path):
                    yield file_path

        return list(_generate_safe_files())


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


class RAGSystem:
    """
    Основная система RAG (Retrieval‑Augmented Generation) для DocAgent‑mini.

    Интегрирует загрузчик документации (DocumentationFileLoader) и читатель
    файлов (DocumentationFileReader) для получения и подготовки данных
    документов. Предоставляет унифицированный интерфейс для последующих
    этапов RAG: поиска релевантных фрагментов и генерации ответов.

    Attributes:
        fileloader (DocumentationFileLoader): Экземпляр загрузчика файлов,
            настроенный с текущими настройками приложения.
        filereader (DocumentationFileReader): Экземпляр читателя файлов,
            используемый для извлечения данных из документов.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует систему RAG с заданными настройками.

        Создаёт экземпляры загрузчика и читателя файлов для последующей работы
        с документацией.

        Args:
            settings (Settings): Конфигурация приложения.
        """
        self.fileloader = DocumentationFileLoader(settings)
        self.filereader = DocumentationFileReader()

    async def get_docs(self) -> List[Path]:
        """
        Асинхронно получает список документов через загрузчик.

        Делегирует загрузку и фильтрацию файлов экземпляру
        DocumentationFileLoader.

        Returns:
            List[Path]: Список путей к валидным файлам документации.

        Raises:
            FileNotFoundError: Если директория с документами не существует.
        """
        return await self.fileloader.get_docs()

    async def get_docs_data(self) -> List[DocumentData]:
        """
        Асинхронно собирает полные данные по всем документам.

        Для каждого документа из списка (полученного через get_docs) выполняет:
        * извлечение метаданных файла;
        * чтение текстового содержимого;
        * разбиение текста на смысловые блоки (чанки).

        Объединяет все данные в структуру DocumentData и возвращает
        список объектов для дальнейшего использования в пайплайне RAG.

        Process:
            1. Получает список путей к документам через self.get_docs().
            2. Для каждого файла вызывает self.filereader.get_file_data(),
               который возвращает объект DocumentData.
            3. Собирает все объекты в единый список.

        Returns:
            List[DocumentData]: Список объектов с полными данными
                по всем документам, включая:
                * метаданные (имя, тип, время создания/изменения, размер);
                * исходное текстовое содержимое;
                * разбиение на смысловые блоки (чанки).

        Raises:
            FileNotFoundError: Если директория с документами не существует
                (передаётся из get_docs).
            IOError: Если возникает ошибка чтения какого‑либо файла.
        """
        return [
            await self.filereader.get_file_data(doc)
            for doc in await self.get_docs()
        ]
