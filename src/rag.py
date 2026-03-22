"""
Модуль src.rag.py — реализация системы RAG (Retrieval‑Augmented Generation)
для DocAgent‑mini.

Содержит компоненты для загрузки и фильтрации документации:
* DocumentationFileLoader — загрузчик файлов с фильтрацией по расширениям
  и проверкой безопасности путей;
* RAGSystem — основная система RAG, интегрирующая загрузчик документов.
"""
# добавить эмбеддинги файлов

import re
from pathlib import Path
from typing import List

from src.settings import Settings


class DocumentationFileLoader:
    """
    Загрузчик документации с фильтрацией файлов и проверкой безопасности путей.


    Обеспечивает безопасную загрузку документов из заданной директории,
    применяя фильтры по расширениям и проверяя, что запрашиваемые файлы
    находятся внутри разрешённой директории.


    Attributes:
        settings (Settings): Настройки приложения, содержащие параметры
            загрузки (путь к документам, шаблон разрешённых имён файлов).
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует загрузчик с заданными настройками.

        Args:
            settings (Settings): Конфигурация приложения (путь к документам,
                шаблон разрешённых имён файлов и т. д.).
        """
        self.settings = settings

    def is_filename_allowed(self, filename: str) -> bool:
        """
        Проверяет, соответствует ли имя файла разрешённому шаблону.

        Использует регулярное выражение из настроек (ALLOWED_FILENAME_PATTERN)
        для фильтрации файлов по расширению.

        Args:
            filename (str): Имя файла для проверки.

        Returns:
            bool: True, если имя файла соответствует шаблону, иначе False.
        """
        if re.match(self.settings.ALLOWED_FILENAME_PATTERN, filename):
            return True
        return False

    def get_file_path(self, filename: str) -> Path:
        """
        Формирует полный безопасный путь к файлу на основе имени.

        Разрешает путь относительно базовой директории документов (DOC_PATH),
        гарантируя нормализацию и разрешение символических ссылок.

        Args:
            filename (str): Имя файла.

        Returns:
            Path: Объект Path, представляющий полный путь к файлу.
        """
        base_dir = Path(self.settings.DOC_PATH).resolve()
        target_path = (base_dir / filename).resolve()
        return target_path

    def is_filepath_safe(self, filepath: Path) -> bool:
        """
        Проверяет безопасность пути к файлу.

        Гарантирует, что запрашиваемый файл находится внутри базовой
        директории документов (DOC_PATH), предотвращая доступ к файлам
        вне этой директории.

        Args:
            filepath (Path): Путь к файлу для проверки.

        Returns:
            bool: True, если путь безопасен (находится внутри DOC_PATH),
                иначе False.
        """
        base_dir = Path(self.settings.DOC_PATH).resolve()
        return str(filepath).startswith(str(base_dir))

    async def get_docs(self) -> List[Path]:
        """
        Асинхронно загружает список безопасных файлов документации.

        Перебирает файлы в директории DOC_PATH, фильтрует их по расширению
        и проверяет безопасность пути. Возвращает список объектов Path
        для валидных файлов.

        Returns:
            List[Path]: Список путей к файлам документации,
                соответствующим критериям фильтрации и безопасности.

        Raises:
            FileNotFoundError: Если директория DOC_PATH не существует.
        """
        def _generate_safe_files():
            dir_path = Path(self.settings.DOC_PATH)

            if not dir_path.exists():
                raise FileNotFoundError(f"Directory not found: {dir_path}")

            for obj in dir_path.iterdir():
                if not obj.is_file():
                    continue

                filename = obj.name
                if not self.is_filename_allowed(filename):
                    continue

                file_path = self.get_file_path(filename)
                if file_path and self.is_filepath_safe(file_path):
                    yield file_path

        return list(_generate_safe_files())


class RAGSystem:
    """
    Основная система RAG (Retrieval‑Augmented Generation) для DocAgent‑mini.

    Интегрирует загрузчик документации (DocumentationFileLoader) и
    предоставляет интерфейс для получения документов, которые могут
    использоваться в последующих этапах RAG (поиск, генерация ответов).

    Attributes:
        fileloader (DocumentationFileLoader): Экземпляр загрузчика файлов,
            настроенный с текущими настройками приложения.
    """

    def __init__(self, settings: Settings):
        """
        Инициализирует систему RAG с заданными настройками.

        Создаёт экземпляр загрузчика файлов для последующей работы
        с документацией.

        Args:
            settings (Settings): Конфигурация приложения.
        """
        self.fileloader = DocumentationFileLoader(settings)

    async def get_docs(self) -> List[Path]:
        """
        Асинхронно получает список документов через загрузчик.

        Делегирует загрузку и фильтрацию файлов экземпляру
        DocumentationFileLoader.

        Returns:
            List[Path]: Список путей к валидным файлам документации.

        Raises:
            FileNotFoundError: Если директория с документами не существует.
        """
        return await self.fileloader.get_docs()
