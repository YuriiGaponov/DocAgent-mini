"""
Модуль src.rag.vectorDB_manager.py — менеджер векторной базы данных
для системы RAG (Retrieval‑Augmented Generation) в DocAgent‑mini.


Содержит класс VectorDBManager для взаимодействия с векторной БД ChromaDB.
Реализует:
* ленивое создание клиента БД (EphemeralClient);
* настройку функции эмбеддингов на основе модели из настроек;
* получение или создание коллекции документов в БД.


Используется для хранения и поиска векторных представлений документов
(эмбеддингов) в пайплайне RAG. Обеспечивает интеграцию с моделью
эмбеддингов и базовую инициализацию структуры хранения данных.
"""


from chromadb import EphemeralClient
from chromadb.api import ClientAPI
from chromadb.api.models import Collection
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

from src.logger import logger
from src.settings import Settings


class VectorDBManager:
    """
    Менеджер для работы с векторной базой данных ChromaDB.

    Обеспечивает:
    * доступ к клиенту БД (ленивая инициализация);
    * настройку функции генерации эмбеддингов;
    * управление коллекциями документов.
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

    def get_or_create_collection(self) -> Collection:
        """
        Получает существующую или создаёт новую коллекцию в БД.

        Использует имя коллекции из настроек (VECTOR_DB_NAME) и функцию
        эмбеддингов для инициализации. Логирует операции создания.
        """
        logger.debug('Запуск VectorDBManager.get_or_create_collection')
        collection = self.client.get_or_create_collection(
            self.settings.VECTOR_DB_NAME,
            embedding_function=self.embedding_function
        )
        logger.debug(f'Коллекция {collection.name} создана')
        return collection
