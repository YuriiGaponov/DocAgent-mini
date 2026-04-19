"""
Модуль src.agent.agent.py — реализация агента DocAgent‑mini.

Содержит класс DocAgent для обработки пользовательских запросов через граф
LangGraph.
Агент использует LLM (ChatOllama) и инструменты (в т. ч. поиск через RAGSystem)
для поиска информации во внутренней документации.
"""


from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.logger import logger
from src.models import AskRequest, State
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import TOOLS
from src.agent.validators import validate_tool_call
from src.rag.rag_system import RAGSystem
from src.settings import Settings


# Имитация БД для хранения состояний
STATES:  dict[str, State] = {}


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
            workflow.add_node('agent', self.call_model)
            workflow.add_node('tools', self.tool_node)
            workflow.add_node("update", self.update_task_id)
            workflow.add_edge(START, 'agent')

            def route_after_agent(state: State) -> str:
                state = validate_tool_call(state)
                last_message: AIMessage = state.messages[-1]
                if last_message.tool_calls:
                    logger.trace('переход к узлу графа "tools"')
                    return "tools"
                else:
                    logger.trace('переход к узлу графа END')
                    return END

            def route_after_tools(state: State) -> str:
                last_message = state.messages[-1]
                tool_name = last_message.name
                if tool_name == 'create_task_id':
                    return "update"
                else:
                    return 'agent'

            workflow.add_conditional_edges(
                'agent',
                route_after_agent
            )
            workflow.add_conditional_edges(
                'tools',
                route_after_tools
            )
            workflow.add_edge('update', 'agent')
            self._graph = workflow.compile()
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
        return TOOLS(self.rag_system)

    @property
    def tool_node(self) -> ToolNode:
        """
        Создаёт узел инструментов (ToolNode) для графа workflow.

        Использует список доступных инструментов агента.
        """
        return ToolNode(self.tools)

    def update_task_id(self, state: State) -> State:
        """
        Обновляет идентификатор задачи (task_id) в состоянии диалога на основе
        содержимого последнего сообщения.

        Извлекает значение из поля content последнего сообщения в истории
        диалога и присваивает его полю task_id объекта состояния.
        Используется в workflow графа для передачи идентификатора задачи
        между этапами обработки.

        Args:
            state (State): текущее состояние диалога, содержащее:
                - messages: историю сообщений (список объектов Message);
                - task_id: текущий идентификатор задачи
                    (может быть не установлен);
                - другие поля состояния согласно определению класса State.

        Returns:
            State: копия состояния диалога с обновлённым полем task_id.
                В возвращаемом объекте:
                - поле task_id установлено в значение, извлечённое из
                content последнего сообщения;
                - остальные поля состояния сохранены без изменений.
        """
        logger.debug('Запуск DocAgent.update_task_id')
        task_id = int(state.messages[-1].content)
        state.task_id = task_id
        logger.debug(f'обновленное состояние {state}')
        return state

    async def call_model(self, state: State):
        """
        Обрабатывает текущее состояние диалога через LLM.

        Вызывает языковую модель для генерации ответа или вызова инструмента.

        Args:
            state (State): текущее состояние диалога, включающее историю
                сообщений пользователя и ассистента.

        Returns:
            State: обновлённое состояние с добавленным ответом LLM
                в истории сообщений.
        """
        logger.debug('Запуск DocAgent.call_model')
        messages = state.messages
        llm_response = await self.llm.ainvoke(messages)
        state.messages.append(llm_response)
        return state

    def get_initial_state(self, user_id: int) -> State:
        """
        Создаёт начальное состояние диалога на основе запроса пользователя.

        Пытается загрузить сохранённое состояние из хранилища STATES
        по user_id.
        Если состояние не найдено, создаёт новое с системным промтом.

        Args:
            user_id (int): идентификатор пользователя.

        State: начальное состояние диалога, включающее:
            - user_id: идентификатор пользователя;
            - messages: список сообщений, содержащий:
                * SystemMessage с системным промптом (SYSTEM_PROMPT) —
                    правила работы агента и контекст взаимодействия;
                * (при наличии сохранённого состояния) -
                    историю предыдущих сообщений диалога.
        """
        logger.debug('Запуск DocAgent.create_initial_state')
        if str(user_id) in STATES:
            initial_state = STATES[str(user_id)]
        else:
            initial_state = State(
                user_id=user_id,
                messages=[SystemMessage(content=SYSTEM_PROMPT)]
            )
        logger.debug('Получено initial_state')
        return initial_state

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
        initial_state = self.get_initial_state(request_data.user_id)
        initial_state.messages.append(HumanMessage(content=request_data.query))
        final_state = await self.graph.ainvoke(initial_state)
        STATES[str(request_data.user_id)] = State(**final_state)
        response = final_state
        return response
