"""
Модуль src.agent.agent.py — реализация агента DocAgent‑mini.

Содержит класс DocAgent для обработки пользовательских запросов через граф
LangGraph.
Агент использует LLM (ChatOllama) и инструменты (в т. ч. поиск через RAGSystem)
для поиска информации во внутренней документации.
"""


from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.logger import logger
from src.models import AskRequest, State
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import TOOLS
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

    SYSTEM_MESSAGE = ''
    HUMAN_MESSAGE = ''

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
            workflow.add_node('agent', self.call_model)
            workflow.add_node('tools', self.tool_node)
            workflow.add_node("update", self.update_task_id)
            workflow.add_edge(START, 'agent')

            def route_after_agent(state: State) -> str:
                last_message = state.messages[-1]
                logger.trace(f'last_message: {last_message}')
                logger.trace(
                    f'last_message.tool_calls: {last_message.tool_calls}'
                )
                if last_message.tool_calls:
                    logger.trace('переход к узлу графа "tools"')
                    return "tools"
                elif last_message.content and 'name' in last_message.content:
                    logger.trace(f'content {last_message.content}')
                    content = last_message.content
                    if "None" in content:
                        content = content.replace("None", "null")
                        logger.trace(
                            f'замена "None" на "null" в content {content}'
                        )
                    import json
                    tool_data = json.loads(content)
                    name = tool_data["name"]
                    logger.trace(f'tool_data: {tool_data}')
                    logger.trace(f'"name" {name, type(name)}')
                    parameters = tool_data["parameters"]
                    if name == 'create_task_id':
                        parameters['task_id'] = state.task_id
                    logger.trace(
                        f'"parameters" {parameters, type(parameters)}'
                    )
                    id = str(hash(tool_data["name"]))
                    logger.trace(f'"id" {id, type(id)}')
                    from langchain_core.messages import ToolCall
                    tool_call = ToolCall(
                        name=name,
                        args=parameters,
                        id=id,
                        type='tool_call'
                    )
                    logger.trace(f'tool_call: {tool_call}')
                    state.messages[-1].tool_calls.append(tool_call)
                    last_message = state.messages[-1]
                    logger.trace(
                        f'last_message после обработки: {last_message}'
                    )
                    logger.trace('переход к узлу графа "tools"')
                    return "tools"
                else:
                    logger.trace('переход к узлу графа END')
                    return END

            def route_after_tools(state: State) -> str:
                last_message = state.messages[-1]
                logger.trace(f'last_message: {last_message}')
                tool_name = last_message.name
                logger.trace(f'tool_name: {tool_name}')
                if tool_name == 'create_task_id':
                    logger.trace('переход к узлу графа "update"')
                    return "update"
                else:
                    logger.trace('переход к узлу графа "agent"')
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
        logger.trace(f'новый task_id {task_id}, {type(task_id)}')
        state.task_id = task_id
        logger.trace(f'обновленное состояние {state}')
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
        logger.trace(f'получено состояние {state}')
        current_state = f'текущий task_id: {state.task_id}\n'
        DocAgent.SYSTEM_MESSAGE.content = current_state + SYSTEM_PROMPT
        state.messages = [
            DocAgent.SYSTEM_MESSAGE, DocAgent.HUMAN_MESSAGE
        ] + state.messages
        messages = state.messages
        logger.trace(f'запуск LLM с messages: {messages}')
        llm_response = await self.llm.ainvoke(messages)
        logger.trace(f'ответ LLM: {llm_response}')
        state.messages.append(llm_response)
        logger.trace(f'updated_state: {state}')
        return state

    def create_initial_state(self, request_data: AskRequest) -> State:
        """
        Создаёт начальное состояние диалога на основе запроса пользователя.

        Формирует состояние с системным промтом и сообщением пользователя.

        Args:
            request_data (AskRequest): объект с данными запроса,
                содержащий идентификатор пользователя (user_id) и текст
                вопроса (query).

        Returns:
            State: начальное состояние диалога, включающее:
                - user_id: идентификатор пользователя;
                - messages: список из системного промпта и сообщения
                  пользователя.
        """
        logger.debug('Запуск DocAgent.create_initial_state')
        DocAgent.SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_PROMPT)
        DocAgent.HUMAN_MESSAGE = HumanMessage(content=request_data.query)
        if f'{str(request_data.user_id)}' in STATES:
            logger.trace('initial_state есть в БД')
            initial_state = STATES[f'{str(request_data.user_id)}']
            initial_state.messages.append(DocAgent.HUMAN_MESSAGE)
            logger.trace('initial_state получен из БД')
        else:
            logger.trace('initial_state нет в БД')
            initial_state = State(
                user_id=request_data.user_id,
                messages=[
                    DocAgent.SYSTEM_MESSAGE,
                    DocAgent.HUMAN_MESSAGE
                ]
            )
            logger.trace('initial_state создан')
        logger.trace(f'initial_state: {initial_state}')
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
        logger.trace(f'входящие данные: {request_data}')
        initial_state = self.create_initial_state(request_data)
        logger.trace('запуск графа')
        final_state = await self.graph.ainvoke(initial_state)
        logger.trace(f'final_state: {final_state}')
        STATES[str(request_data.user_id)] = State(**final_state)
        logger.trace(f'хранилище состояний {STATES}')
        response = final_state
        return response
