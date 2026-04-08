"""
Модуль src.agent.agent.py — реализация агента DocAgent‑mini.

Содержит класс DocAgent для обработки пользовательских запросов. Агент
классифицирует запросы и маршрутизирует их к соответствующим компонентам
системы (например, к RAG‑системе для поиска ответов по документации).
"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

from src.logger import logger
from src.models import AskRequest, State
from src.settings import Settings
from src.rag.rag_system import RAGSystem


SYSTEM_PROMPT = (
    'Ты - агент поиска информации во внутренней документации.\n'
    'Принимаешь запрос пользователя\n,'
    'вызываешь инструмент поиска контекста в коллекции векторной БД.\n'
    'вызываешь инструмент генерации ответа на основе полученного контекста'
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
    def tools(self) -> list:
        return [self.search]

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.LLM_MODEL,
                temperature=self.settings.LLM_TEMPERATURE
            ).bind_tools(self.tools)
        return self._llm

    @property
    def tool_node(self) -> ToolNode:
        return ToolNode(self.tools)

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            workflow = StateGraph(State)
            workflow.add_node("tools", self.tool_node)
            workflow.add_edge(START, 'tools')
            workflow.add_edge('tools', END)
            self._graph = workflow.compile()
        return self._graph

    @tool
    async def search(self, request: str) -> str:
        """
        Инструмент поиска контекста в векторной базе данных.

        Вызывает RAGSystem для поиска релевантной информации по запросу.

        Args:
            request (str): текстовый запрос пользователя для поиска.

        Returns:
            str: найденный контекст из векторной БД.

        Raises:
            Exception: при ошибках взаимодействия с RAGSystem или векторной БД.
        """
        logger.debug('Запуск DocAgent.search')
        context = await self.rag_system.search(request)
        return context

    async def process_query(self, request_data: AskRequest):

        logger.debug('Запуск DocAgent.process_query')

        logger.trace(f'входящие данные: {request_data}')
        state = State(
            user_id=request_data.user_id,
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=request_data.query)
            ]
        )
        logger.trace(f'сформирован state: {state}')

        final_state = await self.llm.ainvoke(state.messages)
        logger.trace(f'сформирован final_state: {final_state}')
        response = final_state
        return response
