"""
Модуль src.rag.rag_system.py — реализация системы RAG
(Retrieval‑Augmented Generation) для DocAgent‑mini.

Содержит основную бизнес‑логику работы с документами в рамках пайплайна RAG:
* загрузку файлов через DocumentationFileLoader (с фильтрацией по расширениям
  и проверкой безопасности путей);
* чтение и извлечение данных через DocumentationFileReader (метаданные,
  текстовое содержимое, разбиение на чанки);
* интеграцию компонентов в рамках класса RAGSystem, предоставляющего
  унифицированный интерфейс для получения подготовленных данных документов.

Ключевые возможности модуля:
* асинхронное получение списка валидных документов (get_docs);
* сбор полных данных по документам в формате DocumentData (get_docs_data),
  включая метаданные, исходное содержимое и разбиение на смысловые блоки.


Используемые компоненты из других модулей:
* src.models.DocumentData — структура данных для представления полного
  описания документа (метаданные + содержимое + чанки);
* src.settings.Settings — конфигурация приложения (пути, шаблоны фильтров);
* src.rag.loader.DocumentationFileLoader — компонент для безопасной
  загрузки и фильтрации файлов;
* src.rag.loader.EmbeddingService — компонент для работы с эмбеддингами;
* src.rag.reader.DocumentationFileReader — компонент для чтения
  файлов и извлечения данных.

Модуль служит центральным звеном в подготовке данных для последующих этапов
RAG: поиска релевантных фрагментов и генерации ответов на основе
документации.
"""

from pathlib import Path
from typing import List

from src.models import DocumentData
from src.settings import Settings
from src.logger import logger
from src.rag.embedding_manager import EmbeddingService
from src.rag.loader import DocumentationFileLoader
from src.rag.reader import DocumentationFileReader


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
        self.embedder = EmbeddingService()
        logger.debug(
            f'\n'
            f'Инициализирована RAG-система: {self.__class__}\n'
            f'Загрузчик документации: {self.fileloader.__class__}\n'
            f'Читатель файлов: {self.filereader.__class__}'
        )

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
        logger.debug('Запуск RAGSystem.get_docs - получение документов.')
        return await self.fileloader.get_docs()

    async def generate_embedding(self, text: str) -> str:
        """Превращает текст в векторы."""
        logger.debug(
            'Запуск RAGSystem.generate_embedding - превратить текст в векторы.'
        )
        embedding = self.embedder.generate_embedding(text)
        logger.debug('Эмбеддинг создан')
        return embedding

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
        logger.debug(
            'Запуск RAGSystem.get_docs_data - получение данных из документов.'
        )
        docs_data = [
            await self.filereader.get_file_data(doc)
            for doc in await self.get_docs()
        ]
        logger.debug(f'Подготовлены данные {len(docs_data)} документов')
        return docs_data
