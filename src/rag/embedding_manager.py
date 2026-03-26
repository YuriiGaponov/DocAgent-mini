"""
Модуль src.rag.embedding_manager.py — сервис генерации эмбеддингов
для системы RAG в DocAgent‑mini.

Содержит класс EmbeddingService для преобразования текста в векторные
представления (эмбеддинги). Реализует:
* ленивую загрузку модели (при первом обращении);
* кэширование модели для повторного использования;
* генерацию эмбеддингов для текстовых фрагментов (одиночных
  и пакетных).

Особенности:
* использует sentence_transformers для работы с моделями;
* загружает модель из настроек приложения (EMBEDDING_MODEL);
* оптимизирует производительность через кэширование;
* логирует операции через встроенный логгер.

Пример использования:
    settings = Settings()
    embedding_service = EmbeddingService(settings)
    # Одиночный эмбеддинг
    embedding = await embedding_service.generate_embedding(
        "Пример текста"
    )
    # Пакетная обработка
    embeddings = await embedding_service.generate_embedding([
        "Текст 1", "Текст 2"
    ])
"""

from typing import List, Union
from sentence_transformers import SentenceTransformer

from src.settings import Settings
from src.logger import logger


class EmbeddingService:
    """
    Сервис генерации эмбеддингов для RAG‑системы.

    Преобразует текст в векторные представления с использованием
    предобученных языковых моделей. Поддерживает одиночную и пакетную
    обработку текстов. Оптимизирует производительность за счёт ленивой
    загрузки и кэширования модели.
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

    async def generate_embedding(
        self,
        text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Асинхронно генерирует эмбеддинг для текста или списка текстов.

        Поддерживает два режима:
        * одиночная обработка — если передан str, возвращает List[float];
        * пакетная обработка — если передан List[str], возвращает
          List[List[float]].

        Логирует количество обрабатываемых текстов и факт завершения операции.
        """
        count = len(text) if isinstance(text, list) else 1
        logger.debug(f"Запуск generate_embedding для {count} текстов")

        if isinstance(text, list):
            # Пакетная обработка: кодируем все тексты сразу
            embeddings = self.embedding_function.encode(text)
            result = embeddings.tolist()
        else:
            # Одиночная обработка: оборачиваем текст в список для encode,
            # берём первый элемент результата
            embedding = self.embedding_function.encode([text])[0]
            result = embedding.tolist()

        logger.debug("Эмбеддинги сгенерированы")
        return result
