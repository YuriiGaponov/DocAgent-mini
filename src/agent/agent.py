"""
Модуль src.agent.agent.py — реализация агента DocAgent‑mini.

Содержит класс DocAgent для обработки пользовательских запросов. Агент
классифицирует запросы и маршрутизирует их к соответствующим компонентам
системы (например, к RAG‑системе для поиска ответов по документации).
"""


from src.logger import logger
from src.models import AskRequest
from src.settings import Settings
from src.rag.rag_system import RAGSystem


class DocAgent:
    """
    Основной агент приложения DocAgent‑mini для обработки запросов
    пользователя.

    Классифицирует входящие запросы и направляет их к соответствующим
    компонентам системы. В текущей реализации поддерживает обработку
    RAG‑запросов через интеграцию с RAGSystem.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует агента с заданными настройками приложения.
        """
        self.settings = settings
        self._rag_sys = None
        logger.debug('Агент инициализирован')

    @property
    def rag_system(self):
        """
        Лениво инициализирует и возвращает экземпляр RAGSystem.

        Создаёт RAGSystem только при первом обращении, используя настройки
        агента. Обеспечивает однократное создание экземпляра.
        """
        if self._rag_sys is None:
            self._rag_sys = RAGSystem(self.settings)
        return self._rag_sys

    def _query_classify(self, query: str) -> str:
        """
        Классифицирует тип пользовательского запроса.

        В текущей реализации всегда возвращает тип RAG_QUERY.
        В будущем может быть расширен для поддержки других типов запросов
        (создание задач, добавление комментариев и т. д.).
        """
        query_type = self.settings.AGENT_QUERY_TYPE.RAG_QUERY
        logger.info(f'Тип запроса: {query_type}')
        return query_type

    async def process_query(self, request_data: AskRequest):
        """
        Обрабатывает входящий пользовательский запрос.

        Выполняет:
        1. Классификацию типа запроса через _query_classify.
        2. Маршрутизацию к соответствующему обработчику (в текущей
           реализации — только RAG‑запросы).
        3. Получение ответа от RAG‑системы.
        """
        logger.debug('Запуск DocAgent.process_query')
        query_type = self._query_classify(request_data.query)

        # === Ответ на запрос ===
        if query_type == self.settings.AGENT_QUERY_TYPE.RAG_QUERY:
            response = await self.rag_system.ask(request_data)
        # === Создание задачи ===
        # === Добавление комментария ===

        return response
