"""
Модуль src.rag.vectorDB_manager.py — менеджер векторной базы данных
для системы RAG (Retrieval‑Augmented Generation) в DocAgent‑mini.

Содержит класс VectorDBManager для взаимодействия с векторной БД ChromaDB.
Реализует:
* ленивое создание клиента БД (EphemeralClient);
* настройку функции эмбеддингов на основе модели из настроек;
* получение или создание коллекции документов в БД;
* добавление документов с эмбеддингами в коллекцию.

Используется для хранения и поиска векторных представлений документов
(эмбеддингов) в пайплайне RAG. Обеспечивает интеграцию с моделью
эмбеддингов и управление структурой хранения данных.
"""

from typing import List

from chromadb import EphemeralClient
from chromadb.api import ClientAPI
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

from src.logger import logger
from src.models import EmbeddedDocument
from src.settings import Settings


class VectorDBManager:
    """
    Менеджер для работы с векторной базой данных ChromaDB.

    Обеспечивает:
    * доступ к клиенту БД (ленивая инициализация);
    * настройку функции генерации эмбеддингов;
    * управление коллекциями документов;
    * загрузку данных (документы, эмбеддинги, метаданные) в коллекцию.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует менеджер векторной БД с заданными настройками.
        """
        self._client: None | ClientAPI = None
        self.settings = settings
        self._embedding_function = None

    @property
    def client(self) -> ClientAPI:
        """
        Лениво создаёт и возвращает клиент ChromaDB (EphemeralClient).
        """
        if self._client is None:
            return EphemeralClient()
        return self._client

    @property
    def embedding_function(self):
        """
        Лениво настраивает и возвращает функцию генерации эмбеддингов.

        Использует модель из настроек приложения (EMBEDDING_MODEL) через
        SentenceTransformerEmbeddingFunction.
        """
        if self._embedding_function is None:
            self._ef = SentenceTransformerEmbeddingFunction(
                self.settings.EMBEDDING_MODEL
            )
        return self._embedding_function

    def get_or_create_collection(self, docs: List[EmbeddedDocument]) -> None:
        """
        Получает существующую или создаёт новую коллекцию в БД и загружает
        в неё документы.

        Выполняет:
        * создание/получение коллекции по имени из настроек (VECTOR_DB_NAME);
        * настройку функции эмбеддингов;
        * добавление документов: чанков, эмбеддингов, метаданных, ID.

        Логирует операции создания коллекции и загрузки данных.
        """
        logger.debug('Запуск VectorDBManager.get_or_create_collection')
        collection = self.client.get_or_create_collection(
            self.settings.VECTOR_DB_NAME,
            embedding_function=self.embedding_function
        )
        logger.debug(f'Коллекция {collection.name} создана/получена')

        for doc in docs:
            collection.add(
                ids=doc.hash_ids,
                embeddings=doc.text_embeddings,
                metadatas=doc.file_metadata,
                documents=doc.chunks
            )
            logger.debug(
                f'Добавлен документ {doc.file_metadata["name"]}: '
                f'{len(doc.chunks)} чанков'
            )

        logger.debug(
            f'Загрузка завершена: обработано {len(docs)} документов'
        )
        return collection
