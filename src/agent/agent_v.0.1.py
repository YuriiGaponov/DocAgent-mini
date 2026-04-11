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
    'Ты - агент работы с внутренней документацией.\n'
    'Внутренняя документация находится в коллекции векторной базы данных.\n'
    'Ты используешь доступные инструменты чтобы:\n'
    'искать информацию во внутренней документации - search\n'
    'создавать задачи - create_task\n'
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
        return [self.search, self.create_task]

    @property
    def tool_node(self) -> ToolNode:
        return ToolNode(self.tools)

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.LLM_MODEL,
                temperature=self.settings.LLM_TEMPERATURE
            ).bind_tools(self.tools)
        return self._llm

    @property
    def graph(self) -> CompiledStateGraph:
        if self._graph is None:
            workflow = StateGraph(State)
            workflow.add_node("agent", self.call_agent)
            workflow.add_node("tools", self.tool_node)
            workflow.add_edge(START, "agent")

            def route_after_agent(state: State) -> str:
                last_message = state.messages[-1]
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    return "tools"
                else:
                    return END
            workflow.add_conditional_edges(
                "agent",
                route_after_agent
            )
            workflow.add_edge("tools", "agent")
            workflow.add_edge("agent", END)
            self._graph = workflow.compile()
        return self._graph

    # async def call_model(self, state: State):
    #     logger.debug('Запуск DocAgent.call_model')
    #     messages = state.messages
    #     updated_state = state
    #     updated_state.messages += [await self.llm.ainvoke(messages)]
    #     return updated_state
    async def call_agent(self, initial_state: State, user_query: str):
        logger.debug('Запуск DocAgent.call_model')
        updated_state = initial_state.model_copy()
        logger.trace(f'получено состояние {updated_state}')
        system = (
            f'Текущий task_id: {updated_state.task_id or 'не задан'}\n'
            'Если задача не создана — сначала вызови create_task\n'
            'Не выдумывай ID'
        )
        updated_state.messages = [
            SystemMessage(content=SYSTEM_PROMPT + system),
            HumanMessage(content=user_query)
        ]
        logger.trace(f'состояние перед запуском модели {updated_state}')
        messages = updated_state.messages
        updated_state.messages += [await self.llm.ainvoke(messages)]
        return updated_state

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

    @tool
    async def create_task(self, task_id: int | None = None) -> str:
        """
        Создаёт новую задачу или генерирует следующий идентификатор задачи.

        Если передан существующий task_id, увеличивает его на 1.
        Если task_id не указан (None), присваивает значение 1.

        Args:
            task_id (int | None): текущий идентификатор задачи. По умолчанию — None.

        Returns:
            str: строковое представление нового идентификатора задачи.
        """
        logger.debug('Запуск DocAgent.create_task')
        if task_id:
            task_id += 1
        else:
            task_id = 1
        logger.debug(f'Создан task_id: {task_id}')
        return task_id

    async def process_query(self, request_data: AskRequest):

        logger.debug('Запуск DocAgent.process_query')

        logger.trace(f'входящие данные: {request_data}')

        system = (
            f'Текущий task_id: {updated_state.task_id or 'не задан'}\n'
            'Если задача не создана — сначала вызови create_task\n'
            'Не выдумывай ID'
        )

        initial_state = State(
            user_id=request_data.user_id,
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=request_data.query)
            ]
        )
        # initial_state = State(
        #     user_id=request_data.user_id
        # )
        # user_query = request_data.query
        logger.trace(f'сформирован initial_state: {initial_state}')

        final_state = await self.graph.ainvoke(initial_state)
        # final_state = await self.call_agent(initial_state, user_query)
        logger.trace(f'сформирован final_state: {final_state}')
        response = final_state
        return response
