"""
Модуль src.rag.rag_system.py — реализация системы RAG
(Retrieval‑Augmented Generation) для DocAgent‑mini.


Предоставляет высокоуровневый интерфейс для подготовки данных
и работы с векторной БД в рамках RAG‑пайплайна.
"""


from flashrank import Ranker, RerankRequest
from src.settings import Settings
from src.logger import logger
from src.rag.collection_initiator import CollectionInitiator


class RAGSystem:
    """
    Основная система RAG для DocAgent‑mini.

    Оркестрирует взаимодействие компонентов пайплайна и обеспечивает
    подготовку данных для векторной БД. Использует векторную БД для поиска
    релевантных фрагментов и FlashRank для реранкинга результатов.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует систему RAG с заданными настройками.

        Создаёт:
        - инициатор коллекции для работы с векторной БД;
        - экземпляр реранкера FlashRank;
        - ссылку на коллекцию документов.
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

        Returns:
            dict: словарь с результатами операции, содержащий:
                - 'status' (str): статус выполнения ('success' или 'error');
                - 'message' (str): описание результата операции.

        Raises:
            Exception: при ошибках создания коллекции (например,
                проблемах подключения к векторной БД).
        """
        logger.debug('Запуск RAGSystem.initiate_collection.')
        return await self.initiator.create_docs_collection()

    async def search(self, request: str) -> str:
        """
        Обрабатывает пользовательский запрос через пайплайн RAG.

        Выполняет:
        * поиск релевантных фрагментов в векторной БД;
        * реранкинг результатов с помощью FlashRank;
        * возврат наиболее релевантного контекста.

        Args:
            request (str): текстовый запрос пользователя для поиска.

        Returns:
            str: наиболее релевантный текстовый фрагмент из векторной БД
                после реранкинга.

        Raises:
            Exception: при ошибках взаимодействия с векторной БД
            или реранкером.

        Workflow:
            1. Запрос отправляется в векторную БД для семантического поиска.
            2. Найденные документы передаются в FlashRank для ранжирования.
            3. Возвращается лучший результат (первый элемент после реранкинга).
        """
        logger.debug('Запуск RAGSystem.search')

        # === Поиск в векторной БД ===
        question = request
        result = self.collection.query(
            query_texts=[question],  # исправлено: ожидается список строк
            n_results=5  # явно задаём количество результатов для реранкинга
        )
        logger.debug('Получен контент из векторной БД.')

        # Проверяем, что результаты найдены
        if not result['documents'] or not result['documents'][0]:
            logger.warning('Не найдено релевантных документов для запроса.')
            return (
                "Не удалось найти релевантную информацию для ответа на запрос."
            )

        # === Реранкинг ===
        passages = [{'text': text} for text in result['documents'][0]]
        rerank_request = RerankRequest(question, passages)
        reranked_results = self.ranker.rerank(rerank_request)

        if reranked_results:
            context = reranked_results[0]['text']
            logger.debug('Выполнен реранкинг, выбран лучший фрагмент.')
        else:
            logger.warning('Реранкинг не вернул результатов.')
            context = "Не удалось определить наиболее релевантный контекст."

        logger.debug('Выполнен реранкинг.')
        return context
