"""
Модуль src.rag.embedding_manager.py — сервис генерации эмбеддингов
для системы RAG в DocAgent‑mini.

Содержит класс EmbeddingService для преобразования текста в векторные
представления (эмбеддинги). Реализует:
* ленивую загрузку модели (при первом обращении);
* кэширование модели для повторного использования;
* генерацию эмбеддингов для текстовых фрагментов.

Особенности:
* использует sentence_transformers для работы с моделями;
* загружает модель из настроек приложения (EMBEDDING_MODEL);
* оптимизирует производительность через кэширование;
* логирует операции через встроенный логгер.

Пример использования:
    settings = Settings()
    embedding_service = EmbeddingService(settings)
    embedding = await embedding_service.generate_embedding("Пример текста")
"""

from typing import List
from sentence_transformers import SentenceTransformer

from src.settings import Settings
from src.logger import logger


class EmbeddingService:
    """
    Сервис генерации эмбеддингов для RAG‑системы.

    Преобразует текст в векторные представления с использованием
    предобученных языковых моделей. Оптимизирует производительность
    за счёт ленивой загрузки и кэширования модели.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует сервис с заданными настройками.
        """
        self.settings = settings
        self._embedding_model = None  # Кэш модели

    @property
    def embedding_function(self) -> SentenceTransformer:
        """
        Лениво загружает и возвращает модель эмбеддингов.

        При первом вызове загружает модель из настроек, кэширует.
        В последующие вызовы возвращает кэшированную модель.
        """
        if self._embedding_model is None:
            logger.debug('Загрузка модели эмбеддингов...')
            self._embedding_model = SentenceTransformer(
                self.settings.EMBEDDING_MODEL
            )
            logger.debug('Модель эмбеддингов загружена и кэширована')
        return self._embedding_model

    def generate_embedding(self, text: str) -> List[float]:
        """
        Асинхронно генерирует эмбеддинг для текста.

        Преобразует входной текст в вектор фиксированной размерности
        с помощью загруженной модели. Логирует начало и завершение
        операции.
        """
        logger.debug('Запуск EmbeddingService.generate_embedding')
        embedded_text = self.embedding_function.encode(text)
        logger.debug('Эмбеддинг сгенерирован')
        return embedded_text
