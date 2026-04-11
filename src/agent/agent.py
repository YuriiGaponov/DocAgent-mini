from src.logger import logger
from src.models import AskRequest
from src.settings import Settings


class DocAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        logger.debug('Агент инициализирован')

    async def process_query(self, request_data: AskRequest):
        logger.debug('Запуск DocAgent.process_query')
        logger.trace(f'входящие данные: {request_data}')
        return request_data
