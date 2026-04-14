"""
Модуль src.agent.agent.py — реализация агента DocAgent‑mini.

Содержит класс DocAgent для обработки пользовательских запросов через граф
LangGraph.
Агент использует LLM (ChatOllama) и инструменты (в т. ч. поиск через RAGSystem)
для поиска информации во внутренней документации.
"""

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


def create_search_tool(settings: Settings):
    """
    Создаёт инструмент поиска для интеграции с LangGraph.

    Возвращает асинхронную функцию search, настроенную на работу
    с RAGSystem для конкретного экземпляра настроек.

    Args:
        settings (Settings): настройки приложения, используемые
            для инициализации RAGSystem.
    Returns:
        Callable: инструмент search для использования в графе workflow.
    """
    @tool
    async def search(request: str) -> str:
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
        logger.debug('Запуск search')
        rag_system = RAGSystem(settings)
        context = await rag_system.search(request)
        logger.debug(f'Получен контекст: {context}')
        return context
    return search


@tool
async def create_task_id(task_id: int | None = None) -> int:
    """
    Создаёт новый идентификатор задачи, увеличивая переданный ID на 1.

    Если task_id не указан (None), начинает нумерацию с 1.
    Если передан task_id, возвращает task_id + 1.

    Args:
        task_id (int | None): текущий идентификатор задачи.
        По умолчанию — None.

    Returns:
        str: строковое представление нового идентификатора задачи.
    """
    logger.debug('Запуск create_task_id')
    if task_id is not None:
        task_id = 0
    task_id += 1
    logger.debug(f'Создан task_id: {task_id}')
    return task_id


@tool
async def add_comment(task_id: int, comment: str) -> str:
    """
    Добавляет комментарий к задаче с указанным идентификатором.

    Args:
        task_id (int): идентификатор задачи, к которой добавляется комментарий.
        comment (str): текст комментария.

    Returns:
        str: подтверждение добавления комментария с указанием ID задачи.
    """
    logger.debug('Запуск add_comment')
    result = f'Комментарий "{comment}" добавлен к задаче {task_id}'
    logger.debug(f'Создан комментарий: {result}')
    return result


class DocAgent:
    """
    Основной агент приложения DocAgent‑mini для обработки запросов
    пользователя.

    Использует LangGraph для оркестрации workflow:
    - вызывает LLM (ChatOllama) для принятия решений;
    - маршрутизирует запросы к инструментам (например, search);
    - обрабатывает диалоги через состояние State.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует агента с заданными настройками приложения.
        """
        self.settings = settings
        self._llm = None
        self._graph = None
        self._rag_sys = None
        logger.debug('Агент инициализирован')

    @property
    def llm(self):
        """
        Лениво инициализирует и возвращает экземпляр языковой модели
        ChatOllama.

        Настраивает модель согласно параметрам из настроек приложения и
        привязывает доступные инструменты (tools).
        """
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.LLM_MODEL,
                temperature=self.settings.LLM_TEMPERATURE
            ).bind_tools(self.tools)
        logger.trace(f'используется LLM: {self._llm}')
        return self._llm

    @property
    def graph(self) -> CompiledStateGraph:
        """
        Лениво инициализирует и компилирует граф workflow (StateGraph).

        Определяет последовательность выполнения узлов:
        1. START → model (вызов LLM);
        2. model → tools (если LLM запросил инструмент);
        3. tools → model (возвращение к LLM после вызова инструмента);
        4. Завершение при отсутствии вызовов инструментов.

        Returns:
            CompiledStateGraph: скомпилированный граф workflow для обработки
            запросов.
        """
        if self._graph is None:
            workflow = StateGraph(State)
            workflow.add_node('model', self.call_model)
            workflow.add_node('tools', self.tool_node)
            workflow.add_node("update", self.update_task_id)
            workflow.add_edge(START, 'model')

            def route_after_agent(state: State) -> str:
                last_message = state.messages[-1]
                if (
                    hasattr(last_message, "tool_calls")
                    and last_message.tool_calls
                ):
                    return "tools"
                else:
                    return END
            workflow.add_conditional_edges(
                'model',
                route_after_agent
            )
            workflow.add_edge('tools', 'model')
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
        """
        Возвращает список инструментов, доступных агенту.

        В текущей реализации включает инструмент поиска, созданный
        через create_search_tool с настройками агента.
        """
        return [create_search_tool(self.settings), create_task_id]

    @property
    def tool_node(self) -> ToolNode:
        """
        Создаёт узел инструментов (ToolNode) для графа workflow.

        Использует список доступных инструментов агента.
        """
        return ToolNode(self.tools)

    def update_task_id(self, state: State) -> State:
        last_message = state.messages[-1].content
        state.task_id = last_message
        return state

    async def call_model(self, state: State):
        """
        Обрабатывает текущее состояние диалога через LLM.

        Добавляет системный промпт и вызывает LLM для генерации ответа
        или вызова инструмента.

        Args:
            state (State): текущее состояние диалога.

        Returns:
            State: обновлённое состояние с ответом LLM.
        """
        logger.debug('Запуск DocAgent.call_model')
        SYSTEM_PROMPT = (
            'Ты - агент, выполняющий 2 вида задач:\n'
            'Задача 1.\n'
            'Поиск контекста во внутренней документации через инструмент search, '
            'когда пользователь задает вопрос, последующая генерация короткого ответа из найденного контекста\n'
            'ПРАВИЛА выполнения задачи 1:\n'
            # '- дожидаешься получения контекста из ответа инструмента search, генерируешь из него короткий ответ на вопрос пользователя\n'
            '- если контекст не найден - отвечаешь: НЕТ ИНФОРМАЦИИ\n'
            '- если в контексте нет релевантной информации - отвечаешь: НЕРЕЛЕВАНТНЫЙ КОНТЕКСТ\n'
            # '2. Управление задачами через инструмент update_task_id, '
            # 'когда пользователь просит создать задачу\n'
        )

        system = SystemMessage(content=SYSTEM_PROMPT)
        updated_state = state.model_copy()
        if isinstance(state.messages[-1], HumanMessage):
            updated_state.messages = [system] + state.messages
        logger.trace(f'updated_state до запуска LLM: {updated_state}')
        messages = updated_state.messages
        logger.trace(f'запуск LLM с messages: {messages}')
        llm_response = await self.llm.ainvoke(messages)
        logger.trace(f'ответ LLM: {llm_response}')
        updated_state.messages.append(llm_response)
        logger.trace(f'updated_state: {updated_state}')
        return updated_state

    async def process_query(self, request_data: AskRequest):
        """
        Обрабатывает входящий пользовательский запрос через граф workflow.

        Выполняет:
        1. Создание начального состояния (initial_state) с сообщением
            пользователя.
        2. Запуск графа workflow (graph.ainvoke) для обработки запроса.
        3. Возврат финального состояния (final_state).

        Args:
            request_data (AskRequest): объект с данными запроса,
                включая идентификатор пользователя и текст вопроса.

        Returns:
            State: финальное состояние диалога после обработки запроса.

        Raises:
            Exception: при ошибках обработки запроса (например,
                проблемах взаимодействия с LLM или графом workflow).
        """
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
