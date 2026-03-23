"""
Модуль src.rag.__init__.py — точка входа для пакета RAG
в DocAgent‑mini.


Обеспечивает удобный импорт ключевых компонентов системы RAG
(Retrieval‑Augmented Generation) из подмодулей пакета. Позволяет
пользователям импортировать основные классы напрямую из `src.rag`,
не указывая полные пути к файлам реализации.


Экспортируемые компоненты:
* DocumentationFileLoader — загрузчик файлов с фильтрацией
  по расширениям и проверкой безопасности путей;
* DocumentationFileReader — читатель файлов, извлекающий метаданные
  и содержимое;
* RAGSystem — основная система RAG, интегрирующая загрузчик
  и читатель документов для подготовки данных в пайплайне RAG.

Пример использования:
    from src.rag import RAGSystem, DocumentationFileLoader
    from src.rag import DocumentationFileReader
"""


from src.rag.loader import DocumentationFileLoader
from src.rag.reader import DocumentationFileReader
from src.rag.rag_system import RAGSystem


__all__ = [
    "DocumentationFileLoader",
    "DocumentationFileReader",
    "RAGSystem"
]
