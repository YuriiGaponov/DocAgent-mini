"""
Модуль src.rag.rag_system.py — реализация системы RAG
(Retrieval‑Augmented Generation) для DocAgent‑mini.

Содержит основную бизнес‑логику работы с документами в рамках пайплайна RAG:
* загрузку файлов через DocumentationFileLoader (фильтрация по расширениям,
  проверка безопасности путей);
* чтение и извлечение данных через DocumentationFileReader (метаданные,
  текст, разбиение на чанки);
* генерацию эмбеддингов через EmbeddingService;
* интеграцию компонентов в классе RAGSystem с унифицированным
  интерфейсом.

Ключевые возможности:
* асинхронное получение списка валидных документов (get_docs);
* сбор полных данных по документам (get_docs_data) с метаданными,
  текстом и чанками;
* преобразование текста в векторные представления (generate_embedding).

Используемые компоненты:
* src.models.EmbeddedDocument — структура данных документа
  с эмбеддингами;
* src.settings.Settings — конфигурация приложения;
* src.logger.logger — логгер для отслеживания операций;
* src.rag.embedding_manager.EmbeddingService — сервис эмбеддингов;
* src.rag.loader.DocumentationFileLoader — загрузчик файлов;
* src.rag.reader.DocumentationFileReader — читатель файлов.

Модуль — центральное звено подготовки данных для этапов RAG: поиска
фрагментов и генерации ответов на основе документации.
"""

from pathlib import Path
from typing import List

import asyncio

from src.models import EmbeddedDocument
from src.settings import Settings
from src.logger import logger
from src.rag.embedding_manager import EmbeddingService
from src.rag.loader import DocumentationFileLoader
from src.rag.reader import DocumentationFileReader


class RAGSystem:
    """
    Основная система RAG (Retrieval‑Augmented Generation) для DocAgent‑mini.

    Интегрирует загрузчик, читатель и сервис эмбеддингов для подготовки
    данных документов. Предоставляет интерфейс для поиска фрагментов
    и генерации ответов.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует систему RAG с заданными настройками.

        Создаёт экземпляры загрузчика, читателя и сервиса эмбеддингов.
        """
        self.fileloader = DocumentationFileLoader(settings)
        self.filereader = DocumentationFileReader()
        self.embedder = EmbeddingService(settings)
        logger.debug(
            f'\n'
            f'Инициализирована RAG-система: {self.__class__}\n'
            f'Загрузчик документации: {self.fileloader.__class__}\n'
            f'Читатель файлов: {self.filereader.__class__}\n'
            f'Сервис эмбеддингов: {self.embedder.__class__}'
        )

    async def get_docs(self) -> List[Path]:
        """
        Асинхронно получает список документов через загрузчик.
        Делегирует загрузку и фильтрацию DocumentationFileLoader.
        """
        logger.debug('Запуск RAGSystem.get_docs - получение документов.')
        return await self.fileloader.get_docs()

    def generate_embedding(
            self, text: str | List[str]
    ) -> List[float] | List[List[float]]:
        """Превращает текст в векторы."""
        logger.debug(
            'Запуск RAGSystem.generate_embedding - превратить текст в векторы.'
        )
        embedding = self.embedder.generate_embedding(text)
        logger.debug('Эмбеддинг создан')
        return embedding

    async def get_chunks(self, text: str) -> List[str]:
        """
        Асинхронно разбивает текст на смысловые блоки (чанки) по \n\n.
        """
        logger.debug('Запуск RAGSystem.get_chunks')
        chunks = text.split('\n\n')
        logger.debug('Разбитие на чанки выполнено')
        return chunks

    async def get_docs_data(self) -> List[dict]:
        """
        Асинхронно собирает полные данные по всем документам.

        Для каждого документа:
        * извлекает метаданные;
        * читает содержимое;
        * разбивает на чанки;
        * генерирует эмбеддинги;
        * формирует EmbeddedDocument.

        Возвращает сериализованные данные для пайплайна RAG.
        """
        try:
            emb_docs_data = []
            docs = await self.get_docs()

            read_tasks = [self.filereader.read_file(doc) for doc in docs]
            readed_docs = await asyncio.gather(
                *read_tasks, return_exceptions=True
            )
            for readed_doc in readed_docs:
                chunks = await self.get_chunks(readed_doc.file_text)
                embeddings = self.generate_embedding(chunks)
                embedded_doc = EmbeddedDocument(
                    readed_doc.file_metadata,
                    chunks,
                    embeddings
                )
                emb_docs_data.append(embedded_doc)
            serialized_data = [emb_doc.to_dict() for emb_doc in emb_docs_data]
            logger.debug(
                f'Подготовлены данные {len(serialized_data)} документов'
            )
            return serialized_data
        except Exception as e:
            logger.error(f'Ошибка в get_docs_data: {e}')
            raise
