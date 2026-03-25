"""
Модуль src.rag.embedding_manager.py — сервис генерации эмбеддингов
для системы RAG (Retrieval‑Augmented Generation) в DocAgent‑mini.


Содержит класс EmbeddingService, отвечающий за преобразование текстового
содержимого в векторные представления (эмбеддинги). Реализует:
* ленивую загрузку модели эмбеддингов (загрузка при первом обращении);
* кэширование загруженной модели для повторного использования;
* асинхронную генерацию векторных представлений текста.


Ключевые особенности:
* использование библиотеки sentence_transformers для работы с моделями
  эмбеддингов;
* загрузка модели на основе настроек приложения (EMBEDDING_MODEL);
* оптимизация производительности за счёт кэширования модели;
* логирование операций через встроенный логгер.


Основные сценарии использования:
* инициализация сервиса с настройками приложения;
* получение функции эмбеддинга (автоматическая загрузка модели при
  первом вызове);
* генерация эмбеддингов для отдельных текстовых фрагментов (чанков).

Зависимости:
* sentence_transformers.SentenceTransformer — модель для генерации
  векторных представлений текста;
* src.settings.Settings — конфигурация приложения (путь/название
  модели эмбеддингов);
* src.logger.logger — логгер для отслеживания операций сервиса.

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
    Сервис генерации эмбеддингов для текстовых данных в системе RAG.

    Обеспечивает преобразование текстового содержимого в векторные
    представления с использованием предобученных языковых моделей.
    Реализует ленивую загрузку и кэширование модели для оптимизации
    производительности.

    Attributes:
        settings (Settings): Настройки приложения, содержащие конфигурацию
            сервиса (в т. ч. название модели эмбеддингов).
        _embedding_model (SentenceTransformer | None): Кэш загруженной
            модели эмбеддингов. Изначально None, загружается при первом
            обращении через свойство embedding_function.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует сервис генерации эмбеддингов с заданными настройками.

        Args:
            settings (Settings): Конфигурация приложения, содержащая
                параметры сервиса (например, EMBEDDING_MODEL).
        """
        self.settings = settings
        self._embedding_model = None  # Кэш модели

    @property
    def embedding_function(self) -> SentenceTransformer:
        """
        Лениво загружает и возвращает модель для генерации эмбеддингов.

        При первом обращении загружает модель из настроек (EMBEDDING_MODEL)
        и кэширует её. В последующие вызовы возвращает уже загруженную
        модель.

        Returns:
            SentenceTransformer: Инициализированная модель для кодирования
                текста в векторные представления.
        """
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.settings.EMBEDDING_MODEL
            )
        return self._embedding_model

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Асинхронно генерирует эмбеддинг (векторное представление) для текста.

        Использует загруженную модель эмбеддингов для преобразования
        входного текста в вектор фиксированной размерности. Логирует начало
        операции.

        Args:
            text (str): Текст, который необходимо преобразовать в эмбеддинг.

        Returns:
            List[float]: Векторное представление текста в виде списка
                чисел с плавающей точкой.

        Example:
            >>> embedding = await service.generate_embedding("Hello world")
            >>> len(embedding)
            768  # или другая размерность в зависимости от модели
        """
        logger.debug('Запуск EmbeddingService.generate_embedding')
        embedded_text = self.embedding_function.encode(text)
        logger.debug('Эмбеддинг сгенерирован')
        return embedded_text
