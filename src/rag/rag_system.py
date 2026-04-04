"""
Модуль src.rag.rag_system.py — реализация системы RAG
(Retrieval‑Augmented Generation) для DocAgent‑mini.

Предоставляет высокоуровневый интерфейс для подготовки данных
и работы с векторной БД в рамках RAG‑пайплайна.
"""


import ollama
from flashrank import Ranker, RerankRequest

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
        self.llm_model = settings.LLM_MODEL
        self.initiator = CollectionInitiator(settings)
        self.collection = self.initiator.collection
        self.ranker = Ranker()
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

    async def ask(self, request_data: AskRequest) -> str:
        """
        Обрабатывает пользовательский запрос через пайплайн RAG.

        Выполняет:
        * поиск релевантных фрагментов в векторной БД;
        * реранкинг результатов;
        * генерацию ответа с помощью LLM на основе контекста.
        """
        logger.debug(f'Запуск RAGSystem.ask, request_data: {request_data}')

        # === Поиск в векторной БД ===
        question = request_data.query
        result = self.collection.query(
            query_texts=question
        )
        logger.debug('Получен контент из векторной БД.')

        # === Реранкинг ===
        passages = [{'text': text} for text in result['documents'][0]]
        rerank_request = RerankRequest(question, passages)
        context = self.ranker.rerank(rerank_request)[0]['text']
        logger.debug('Выполнен реранкинг.')

        # === Запрос LLM ===
        prompt = (
            f'Отвечай только на основе следующего контекста.'
            f'Если в контексте нет нужных данных для ответа'
            f'ответь дословно: "Предоставлен нерелевантный контекст"'
            f'НЕ ВКЛЮЧАЙ в ответ данные, полученные не из контекста'
            f'Контекст: {context}'
            f'Вопрос: {question}'
        )
        llm_response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        response = llm_response['message']['content']
        logger.debug('Получен ответ LLM.')

        return response
