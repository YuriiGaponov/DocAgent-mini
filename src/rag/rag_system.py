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
* преобразование текста в векторные представления (generate_embedding);
* создание коллекции документов в векторной БД (create_docs_collection).

Модуль — центральное звено подготовки данных для этапов RAG: поиска
фрагментов и генерации ответов на основе документации.
"""

import hashlib
from pathlib import Path
from typing import List

import asyncio

from src.models import EmbeddedDocument, ReadedDocument
from src.settings import Settings
from src.logger import logger
from src.rag.embedding_manager import EmbeddingService
from src.rag.loader import DocumentationFileLoader
from src.rag.reader import DocumentationFileReader
from src.rag.vectorDB_manager import VectorDBManager


class RAGSystem:
    """
    Основная система RAG (Retrieval‑Augmented Generation) для DocAgent‑mini.

    Интегрирует загрузчик, читатель, сервис эмбеддингов и менеджер
    векторной БД для подготовки данных документов. Предоставляет
    унифицированный интерфейс для поиска фрагментов и генерации ответов.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует систему RAG с заданными настройками.

        Создаёт экземпляры загрузчика, читателя, сервиса эмбеддингов
        и менеджера векторной БД.
        """
        self.fileloader = DocumentationFileLoader(settings)
        self.filereader = DocumentationFileReader()
        self.embedder = EmbeddingService(settings)
        self.client = VectorDBManager(settings)
        logger.debug(f'Инициализирована RAG-система: {self.__class__}')

    async def get_docs(self) -> List[Path]:
        """
        Асинхронно получает список документов через загрузчик.

        Делегирует загрузку и фильтрацию файлов экземпляру
        DocumentationFileLoader.
        """
        logger.debug('Запуск RAGSystem.get_docs — получение документов.')
        return await self.fileloader.get_docs()

    def generate_embedding(
            self, text: str | List[str]
    ) -> List[float] | List[List[float]]:
        """
        Превращает текст в векторные представления (эмбеддинги).

        Использует сервис эмбеддингов для преобразования текстовых данных
        в векторы фиксированной размерности.
        """
        logger.debug(
            'Запуск RAGSystem.generate_embedding — превратить текст в векторы.'
        )
        embedding = self.embedder.generate_embedding(text)
        logger.debug('Эмбеддинг создан')
        return embedding

    def get_chunks(self, text: str) -> List[str]:
        """
        Разбивает текст на смысловые блоки (чанки) по двойным переносам строк.

        Возвращает список строк, где каждый элемент — отдельный чанк текста.
        """
        logger.debug('Запуск RAGSystem.get_chunks')
        chunks = text.split('\n\n')
        logger.debug('Разбиение на чанки выполнено')
        return chunks

    def generate_hash_id(self, content: str) -> str:
        """
        Генерирует короткий хеш‑идентификатор для содержимого.

        Использует SHA‑256 и обрезает результат до 16 символов
        для компактности.
        """
        hash_object = hashlib.sha256(content.encode())
        return hash_object.hexdigest()[:16]

    async def get_docs_data(self) -> List[EmbeddedDocument]:
        """
        Асинхронно собирает полные данные по всем документам.

        Для каждого документа выполняет:
        * извлечение метаданных;
        * чтение текстового содержимого;
        * разбиение текста на чанки;
        * генерацию эмбеддингов для каждого чанка;
        * формирование структуры EmbeddedDocument с уникальными ID.

        Возвращает список объектов EmbeddedDocument для дальнейшего
        использования в пайплайне RAG.
        """
        try:
            emb_docs_data = []
            docs = await self.get_docs()

            read_tasks = [self.filereader.read_file(doc) for doc in docs]
            readed_docs = await asyncio.gather(
                *read_tasks, return_exceptions=True
            )
            # Фильтруем только успешные результаты (объекты ReadedDocument)
            valid_docs = [
                doc for doc in readed_docs
                if isinstance(doc, ReadedDocument)
            ]

            # for readed_doc in readed_docs:
            for readed_doc in valid_docs:
                chunks = self.get_chunks(readed_doc.file_text)
                hash_ids = [self.generate_hash_id(chunk) for chunk in chunks]
                embeddings = self.generate_embedding(chunks)
                # embedded_doc = EmbeddedDocument(
                #     readed_doc.file_metadata,
                #     chunks,
                #     hash_ids,
                #     embeddings
                # )
                embedded_doc = EmbeddedDocument(
                    file_metadata=readed_doc.file_metadata,
                    chunks=chunks,
                    hash_ids=hash_ids,
                    text_embeddings=embeddings
                )
                emb_docs_data.append(embedded_doc)

            logger.debug(
                f'Подготовлены данные для добавления {len(emb_docs_data)} документов\n'
                f'создан экземпляр {emb_docs_data.__class__}'
            )
            return emb_docs_data
        except Exception as e:
            logger.error(f'Ошибка в get_docs_data: {e}')
            raise

    async def create_docs_collection(self):
        """
        Создаёт коллекцию документов в векторной базе данных.

        Выполняет:
        * сбор данных по документам через get_docs_data;
        * создание/получение коллекции в векторной БД через VectorDBManager;
        * загрузку данных (чанки, эмбеддинги, метаданные, ID) в коллекцию.

        Возвращает статус операции и имя созданной коллекции.
        """
        try:
            docs = await self.get_docs_data()
            collection = self.client.get_or_create_collection(docs)
            logger.debug(
                f'Коллекция {collection.name} успешно создана и заполнена'
            )
            return {
                'status': 'success',
                'message': (
                    f'Коллекция {collection.name} создана, '
                    f'создано {collection.count()} записей'
                )
            }
        except Exception as e:
            logger.error(f'Ошибка создания коллекции: {e}')
            raise
