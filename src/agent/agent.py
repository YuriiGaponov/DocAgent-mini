from langchain_ollama import ChatOllama

from src.logger import logger
from src.models import AskRequest
from src.settings import Settings


class DocAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = None
        logger.debug('Агент инициализирован')

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatOllama(
                model=self.settings.LLM_MODEL,
                temperature=self.settings.LLM_TEMPERATURE
            )
        logger.trace(f'используется LLM: {self._llm}')
        return self._llm

    async def process_query(self, request_data: AskRequest):
        logger.debug('Запуск DocAgent.process_query')
        logger.trace(f'входящие данные: {request_data}')
        request = request_data.query
        response = await self.llm.ainvoke(request)
        return response
