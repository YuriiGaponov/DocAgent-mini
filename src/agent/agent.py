"""src.agent.agent.py"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.logger import logger
from src.models import AskRequest, State
from src.rag.rag_system import RAGSystem
from src.settings import Settings


class DocAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = None
        self._graph = None
        self._rag_sys = None
        logger.debug('Агент инициализирован')

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.LLM_MODEL,
                temperature=self.settings.LLM_TEMPERATURE
            ).bind_tools(self.tools)
        logger.trace(f'используется LLM: {self._llm}')
        return self._llm

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            workflow = StateGraph(State)
            workflow.add_node('model', self.call_model)
            workflow.add_node('tools', self.tool_node)
            workflow.add_edge(START, 'model')
            workflow.add_edge('model', END)
            self._graph = workflow.compile()
            logger.trace('граф скомпилирован')
        return self._graph

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
    def tool_node(self) -> ToolNode:
        return ToolNode(self.tools)

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
        logger.debug(f'Получен контекст: {context}')
        return context

    async def call_model(self, state: State):
        logger.debug('Запуск DocAgent.call_model')
        SYSTEM_PROMPT = (
            'Ты - AI-агент, отвечающий на запросы пользователя'
        )
        system = SystemMessage(content=SYSTEM_PROMPT)
        updated_state = state.model_copy()
        updated_state.messages = [system] + state.messages
        logger.trace(f'updated_state до запуска LLM: {updated_state}')
        messages = updated_state.messages
        llm_response = await self.llm.ainvoke(messages)
        logger.trace(f'ответ LLM: {llm_response}')
        updated_state.messages.append(llm_response)
        logger.trace(f'updated_state: {updated_state}')
        return updated_state

    async def process_query(self, request_data: AskRequest):
        logger.debug('Запуск DocAgent.process_query')
        logger.trace(f'входящие данные: {request_data}')
        initial_state = State(
            user_id=request_data.user_id,
            messages=[
                HumanMessage(content=request_data.query)
            ]
        )
        logger.trace(f'initial_state: {initial_state}')
        logger.trace('запуск графа')
        final_state = await self.graph.ainvoke(initial_state)
        logger.trace(f'final_state: {final_state}')
        response = final_state
        return response
