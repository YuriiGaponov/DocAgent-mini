"""src.rag.embedding_manager.py"""


from src.logger import logger


class EmbeddingService:
    async def generate_embedding(self, text: str) -> str:
        """Превращает текст в векторы."""
        logger.debug(
            'Запуск EmbeddingService.generate_embedding'
        )
        return text
