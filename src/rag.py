"""src.rag.py"""
# Загрузите документацию из папки
# Безопасные пути
# Создайте эмбеддинги для документов


import re
from pathlib import Path

from src.settings import Settings


class DocumentationFileLoader:
    def __init__(self, settings: Settings):
        self.settings = settings

    def is_filename_allowed(self, filename: str) -> bool | None:
        if re.match(self.settings.ALLOWED_FILENAME_PATTERN, filename):
            return True
        return False

    def get_file_path(self, filename: str) -> bool | None:
        base_dir = Path(self.settings.DOC_PATH).resolve()
        target_path = (base_dir / filename).resolve()
        return target_path

    def is_filepath_safe(self, filepath: str) -> bool | None:
        base_dir = Path(self.settings.DOC_PATH).resolve()
        if str(filepath).startswith(str(base_dir)):
            return True
        return False

    async def get_docs(self):
        # Папка с доками
        dir_path = Path(self.settings.DOC_PATH)
        # Содержимое папки
        dir_content = dir_path.iterdir()
        # Файлы в папке
        files = [obj for obj in dir_content if obj.is_file()]
        # имена файлов
        all_file_names = [file.name for file in files]
        # допустимые имена файлов
        allowed_file_names = [filename for filename in all_file_names if self.is_filename_allowed(filename)]
        # пути к файлам
        file_pathes = [self.get_file_path(file_name) for file_name in allowed_file_names]
        # безопасные пути к файлам
        safe_file_pathes = [file_path for file_path in file_pathes if self.is_filepath_safe(file_path)]
        return safe_file_pathes


class RAGSystem:
    def __init__(self, settings: Settings):
        self.fileloader = DocumentationFileLoader(settings)

    async def get_docs(self):
        return await self.fileloader.get_docs()
