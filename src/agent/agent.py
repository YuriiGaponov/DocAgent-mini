"""
Модуль src.agent.agent.py — реализация агента DocAgent‑mini.

Содержит класс DocAgent для обработки пользовательских запросов. Агент
классифицирует запросы и маршрутизирует их к соответствующим компонентам
системы (например, к RAG‑системе для поиска ответов по документации).
"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama

from src.logger import logger
from src.models import AskRequest, State
from src.settings import Settings
from src.rag.rag_system import RAGSystem


SYSTEM_PROMPT = (
    'Ты  - агент поиска информации во внутренней документации.\n'
    'Принимаешь запрос пользователя\n,'
    'вызываешь инструмент поиска контекста в коллекции векторной БД.\n'
    'на основе полученного контекста генерируешь ответ'
)


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
        self._llm = None
        self._graph = None
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

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.LLM_MODEL,
                temperature=self.settings.LLM_TEMPERATURE
            )
        return self._llm

    @property
    def graph(self) -> StateGraph:
        if self._graph is None:
            workflow = StateGraph(State)
            workflow.add_node('search', self.search)
            workflow.add_node('response', self.response)
            workflow.add_edge(START, 'search')
            workflow.add_edge('search', 'response')
            workflow.add_edge('response', END)
            self._graph = workflow.compile()
        return self._graph

    @tool
    async def search(self, state: State):
        """Ищет контекст в векторной базе данных."""
        logger.debug('Запуск инструмента DocAgent.search')
        request = state.messages[-1]
        logger.trace(f'Сформирован запрос: {request}')
        logger.trace(f'запрос передается в RAGSystem {self.rag_system}')
        response = self.rag_system.search(request)
        logger.trace(f'получен контекст {response}')
        return response

    @tool
    async def response(self, state: State):
        """Возвращает последнее сообщение в state"""
        return state.messages[-1]

    async def process_query(self, request_data: AskRequest):

        logger.debug('Запуск DocAgent.process_query')

        logger.trace(f'входящие данные: {request_data}')
        initial_state = State(
            user_id=request_data.user_id,
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=request_data.query)
            ]
        )
        logger.trace(f'сформирован initial_state: {initial_state}')
        response = self.llm.invoke(initial_state.messages)

        return response
