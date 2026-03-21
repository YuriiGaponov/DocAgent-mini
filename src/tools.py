"""src.tools.py"""


import re
from pathlib import Path

from src.settings import Settings


def safe_file_name(filename: str, settings: Settings) -> bool | None:
    if re.match(settings.SAFE_FILENAME_PATTERN, filename):
        return True


def file_path(filename: str, settings: Settings) -> bool | None:
    base_dir = Path(settings.DOC_PATH).resolve()
    target_path = (base_dir / filename).resolve()
    return target_path


def safe_file_path(filepath: str, settings: Settings) -> bool | None:
    base_dir = Path(settings.DOC_PATH).resolve()
    if str(filepath).startswith(str(base_dir)):
        return True


async def get_docs(settings: Settings):
    dir_path = Path(settings.DOC_PATH)
    files = [file_path(f.name, settings) for f in dir_path.iterdir() if f.is_file() and safe_file_name(f.name, settings) and safe_file_path(file_path(f.name, settings), settings)]
    return files
