"""
tests.db.test_relational_db

Набор тестов для модуля src.db.relational_db.

Проверяет корректность работы компонентов слоя доступа к реляционной БД:
- формирование URL подключения для разных СУБД;
- обработку ошибок при запросе неподдерживаемой СУБД;
- создание и валидность асинхронных сессий;
- ленивую инициализацию асинхронного движка (кеширование экземпляра).
"""

import pytest
from pathlib import Path

from src.db.relational_db import get_providerDB
from src.settings import Settings


class TestRelationalDB:
    """
    Тестовый класс для проверки функционала модуля src.db.relational_db.

    Покрывает ключевые сценарии:
    - генерацию URL для поддерживаемых СУБД;
    - обработку запросов для неподдерживаемых СУБД;
    - работу асинхронного контекстного менеджера сессий;
    - оптимизацию инициализации движка (единичный экземпляр, ленивая загрузка).

    Используемые фикстуры:
        mock_settings: мок-объект настроек приложения.
    """
    @pytest.mark.parametrize(
        'DBMS, DB_DIR, DB_NAME, expected_url',
        [
            ('sqlite', Path('test_path'), 'test_sqlite.db',
             'sqlite+aiosqlite:///test_path/test_sqlite.db')
        ]
    )
    def test_available_db_url(
        self,
        mock_settings: Settings,
        DBMS: str, DB_DIR: Path, DB_NAME: str, expected_url: Path
    ):
        """
        Проверяет, что get_db_url() корректно формирует URL для поддерживаемой
        СУБД.

        Сценарий:
            1. Настраивает mock_settings с тестовыми значениями.
            2. Создаёт экземпляр providerDB.
            3. Сравнивает фактический URL с ожидаемым.

        Args:
            mock_settings (Settings): мок-объект настроек.
            DBMS (str): тип СУБД (например, 'sqlite').
            DB_DIR (Path): путь к директории БД.
            DB_NAME (str): имя файла БД.
            expected_url (str): ожидаемый URL подключения.

        Assertions:
            - Фактический URL совпадает с ожидаемым.
        """
        mock_settings.DBMS = DBMS
        mock_settings.DB_DIR = DB_DIR
        mock_settings.DB_NAME = DB_NAME
        providerDB = get_providerDB(mock_settings)
        assert providerDB.get_db_url() == expected_url

    def test_get_db_url_unsupported_dbms(self, mock_settings: Settings):
        """
        Проверяет обработку запроса для неподдерживаемой СУБД.

        Сценарий:
            1. Устанавливает unsupported в DBMS mock_settings.
            2. Пытается получить URL через providerDB.
            3. Убеждается, что возникает KeyError
                с указанием неподдерживаемого типа.

        Args:
            mock_settings (Settings): мок-объект настроек.

        Assertions:
            - Возникает KeyError, сообщение содержит 'unsupported'.
        """
        mock_settings.DBMS = "unsupported"
        providerDB = get_providerDB(mock_settings)
        with pytest.raises(KeyError, match="unsupported"):
            providerDB.get_db_url()

    @pytest.mark.anyio
    async def test_get_async_session_creates_valid_session(
        self,
        mock_settings: Settings
    ):
        """
        Проверяет создание валидной асинхронной сессии через
        get_async_session().

        Сценарий:
            1. Создаёт providerDB с mock_settings.
            2. Получает сессию через асинхронный контекстный менеджер.
            3. Проверяет, что сессия не None и имеет метод execute.

        Args:
            mock_settings (Settings): мок-объект настроек.

        Assertions:
            - Сессия не равна None.
            - У сессии есть атрибут execute (базовая проверка валидности).
        """
        providerDB = get_providerDB(mock_settings)
        async for session in providerDB.get_async_session():
            assert session is not None
            assert hasattr(session, "execute")

    def test_lazy_engine_initialization(self, mock_settings: Settings):
        """
        Проверяет ленивую инициализацию и кеширование асинхронного движка.

        Сценарий:
            1. Создаёт providerDB без предварительной инициализации движка.
            2. Убеждается, что _async_engine равен None до первого вызова.
            3. Получает движок через get_async_engine() (первый вызов).
            4. Получает движок повторно (второй вызов).
            5. Сравнивает экземпляры движка: они должны быть идентичны.

        Args:
            mock_settings (Settings): мок-объект настроек.

        Assertions:
            - До первого вызова _async_engine равен None.
            - Первый и второй вызовы get_async_engine()
              возвращают один и тот же объект.
        """
        providerDB = get_providerDB(mock_settings)

        # До первого вызова движка его быть не должно
        assert providerDB._async_engine is None

        # Первый вызов — движок создаётся
        engine1 = providerDB.get_async_engine()
        assert engine1 is not None

        # Второй вызов — возвращается тот же движок
        engine2 = providerDB.get_async_engine()
        assert engine1 is engine2
