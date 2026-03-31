"""
Модуль src.rag.rag_system.py — реализация системы RAG
(Retrieval‑Augmented Generation) для DocAgent‑mini.

Предоставляет высокоуровневый интерфейс для подготовки данных
и работы с векторной БД в рамках RAG‑пайплайна.
"""


from src.settings import Settings
from src.logger import logger
from src.models import AskRequest
from src.rag.collection_initiator import CollectionInitiator


class RAGSystem:
    """
    Основная система RAG для DocAgent‑mini.

    Оркестрирует взаимодействие компонентов пайплайна и обеспечивает
    подготовку данных для векторной БД.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует систему RAG с заданными настройками.
        """
        self.initiator = CollectionInitiator(settings)
        self.collection = self.initiator.collection
        logger.debug(f'Инициализирована RAG-система: {self.__class__}')

    async def initiate_collection(self):
        """
        Асинхронно создаёт коллекцию документов в векторной БД
        через оркестратор.

        Возвращает словарь с полями:
        - 'status': статус выполнения;
        - 'message': описание результата.
        """
        logger.debug('Запуск RAGSystem.initiate_collection.')
        return await self.initiator.create_docs_collection()

    async def ask(self, request_data: AskRequest):
        logger.debug(f'Запуск RAGSystem.ask, request_data: {request_data}')
        result = self.collection.query(
            query_texts=request_data.query,
            n_results=1
        )
        return result
