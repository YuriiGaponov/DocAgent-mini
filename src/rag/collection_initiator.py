"""
Модуль src.rag.collection_initiator.py - инициализации коллекции.

Содержит класс CollectionInitiator для подготовки и загрузки данных документов
в векторную базу данных в рамках RAG‑системы (Retrieval‑Augmented Generation)
проекта DocAgent‑mini.
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
from src.rag.utils import generate_hash_id, get_chunks
from src.rag.vectorDB_manager import VectorDBManager


class CollectionInitiator:
    """
    Инициатор создания коллекции документов для RAG‑системы DocAgent‑mini.

    Интегрирует компоненты пайплайна: загрузчик файлов, читатель,
    сервис эмбеддингов и менеджер векторной БД. Формирует
    структурированные данные (EmbeddedDocument) и загружает их
    в векторную базу для последующего поиска и генерации ответов.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует инициатор коллекции с заданными настройками.

        Создаёт экземпляры зависимых компонентов:
        - загрузчика файлов (DocumentationFileLoader);
        - читателя файлов (DocumentationFileReader);
        - сервиса генерации эмбеддингов (EmbeddingService);
        - менеджера векторной БД (VectorDBManager).
        """
        self.fileloader = DocumentationFileLoader(settings)
        self.filereader = DocumentationFileReader()
        self.embedder = EmbeddingService(settings)
        self.client = VectorDBManager(settings)
        self.collection = self.client.collection
        logger.debug(f'Инициализирована RAG-система: {self.__class__}')

    async def get_docs(self) -> List[Path]:
        """
        Асинхронно получает список путей к документам через загрузчик.

        Делегирует загрузку и фильтрацию файлов экземпляру
        DocumentationFileLoader.
        """

        logger.debug('Запуск CollectionInitiator.get_docs')
        return await self.fileloader.get_docs()

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

            for readed_doc in readed_docs:
                chunks = get_chunks(readed_doc.file_text)
                hash_ids = [generate_hash_id(chunk) for chunk in chunks]
                embeddings = self.embedder.generate_embedding(chunks)
                embedded_doc = EmbeddedDocument(
                    file_metadata=readed_doc.file_metadata,
                    chunks=chunks,
                    hash_ids=hash_ids,
                    text_embeddings=embeddings
                )
                emb_docs_data.append(embedded_doc)

            logger.debug(
                f'Подготовлены данные для добавления {len(emb_docs_data)} '
                f'документов\nсоздан экземпляр {emb_docs_data.__class__}'
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

        Возвращает словарь с:
        - статусом операции ('success' или ошибка);
        - сообщением о результате (имя коллекции и количество записей).
        """
        try:
            docs = await self.get_docs_data()
            self.client.add_docs_to_collection(docs)
            logger.debug(
                f'Коллекция {self.collection.name} успешно создана и заполнена'
            )
            return {
                'status': 'success',
                'message': (
                    f'Коллекция {self.collection.name} создана, '
                    f'создано {self.collection.count()} записей'
                )
            }
        except Exception as e:
            logger.error(f'Ошибка создания коллекции: {e}')
            raise
