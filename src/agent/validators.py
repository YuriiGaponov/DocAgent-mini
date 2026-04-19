"""src.agent.validators.py

Модуль содержит функции для валидации и предварительной обработки сообщений
агента, в частности — для корректного формирования вызовов инструментов
(tool calls) на основе ответов LLM.
"""


import json

from langchain_core.messages import AIMessage, ToolCall
from pydantic import ValidationError

from src.logger import logger
from src.models import State


def validate_tool_call(state: State) -> AIMessage:
    """
    Валидирует и преобразует последнее сообщение состояния в структурированный
    вызов инструмента.

    Выполняет следующие действия:
    1. Проверяет, что последнее сообщение (state.messages[-1]) является
        экземпляром AIMessage.
    2. Если сообщение не является AIMessage, выбрасывает ValidationError.
    3. Если LLM вернула JSON‑описание инструмента в content (вместо ToolCall),
       парсит его и создаёт объект ToolCall.
    4. Для инструмента create_task_id автоматически подставляет текущий
        task_id из состояния.
    5. Добавляет сформированный ToolCall в список tool_calls сообщения.

    Args:
        state (State): текущее состояние диалога, содержащее историю сообщений
            и метаданные (включая task_id).

    Returns:
        AIMessage: последнее сообщение из состояния (state.messages[-1]),
            дополненное объектом ToolCall в поле tool_calls, если
            преобразование было необходимо и успешно выполнено.

    Raises:
        ValidationError: если последнее сообщение не является экземпляром
            AIMessage.
    """
    message = state.messages[-1]
    if not isinstance(message, AIMessage):
        logger.error('LLM вернула сообщение неверного типа')
        raise ValidationError('LLM вернула сообщение неверного типа')
    if (
        not message.tool_calls
        and message.content
        and 'name' in message.content
    ):
        content = message.content
        if "None" in content:
            content = content.replace("None", "null")
        tool_data = json.loads(content)
        name = tool_data["name"]
        parameters = tool_data["parameters"]
        if name == 'create_task_id':
            parameters['task_id'] = state.task_id
        tool_call = ToolCall(
            name=name,
            args=parameters,
            id=str(hash(name)),
            type='tool_call'
        )
        message.tool_calls.append(tool_call)
        logger.debug(f'В AIMessage добавлен tool_calls {message.tool_calls}')
        return state
    else:
        return state
