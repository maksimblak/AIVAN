"""
Базовые классы для обработки документов
"""

from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Базовое исключение для ошибок обработки документов"""
    def __init__(self, message: str, error_code: str = "PROCESSING_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

@dataclass
class DocumentResult:
    """Результат обработки документа"""
    success: bool
    data: Dict[str, Any]
    message: str
    processing_time: float
    timestamp: datetime
    error_code: Optional[str] = None

    @classmethod
    def success_result(cls, data: Dict[str, Any], message: str = "Обработка успешно завершена", processing_time: float = 0.0) -> DocumentResult:
        """Создать успешный результат"""
        return cls(
            success=True,
            data=data,
            message=message,
            processing_time=processing_time,
            timestamp=datetime.now()
        )

    @classmethod
    def error_result(cls, message: str, error_code: str = "PROCESSING_ERROR", processing_time: float = 0.0) -> DocumentResult:
        """Создать результат с ошибкой"""
        return cls(
            success=False,
            data={},
            message=message,
            processing_time=processing_time,
            timestamp=datetime.now(),
            error_code=error_code
        )

class DocumentProcessor(ABC):
    """Базовый абстрактный класс для обработчиков документов"""

    def __init__(self, name: str, max_file_size: int = 50 * 1024 * 1024):  # 50MB по умолчанию
        self.name = name
        self.max_file_size = max_file_size
        self.supported_formats: List[str] = []

    @abstractmethod
    async def process(self, file_path: Union[str, Path], **kwargs) -> DocumentResult:
        """Абстрактный метод для обработки документа"""
        pass

    def validate_file(self, file_path: Union[str, Path]) -> tuple[bool, str]:
        """Валидация файла перед обработкой"""
        path = Path(file_path)

        # Проверка существования файла
        if not path.exists():
            return False, f"Файл не найден: {file_path}"

        # Проверка размера файла
        file_size = path.stat().st_size
        if file_size > self.max_file_size:
            max_mb = self.max_file_size / (1024 * 1024)
            current_mb = file_size / (1024 * 1024)
            return False, f"Размер файла ({current_mb:.1f} МБ) превышает максимальный ({max_mb:.1f} МБ)"

        # Проверка формата файла
        file_ext = path.suffix.lower()
        if self.supported_formats and file_ext not in self.supported_formats:
            return False, f"Неподдерживаемый формат файла: {file_ext}. Поддерживаемые: {', '.join(self.supported_formats)}"

        return True, "OK"

    async def safe_process(self, file_path: Union[str, Path], **kwargs) -> DocumentResult:
        """Безопасная обработка документа с валидацией и обработкой ошибок"""
        start_time = asyncio.get_event_loop().time()

        try:
            # Валидация файла
            is_valid, validation_message = self.validate_file(file_path)
            if not is_valid:
                processing_time = asyncio.get_event_loop().time() - start_time
                return DocumentResult.error_result(
                    message=validation_message,
                    error_code="VALIDATION_ERROR",
                    processing_time=processing_time
                )

            # Обработка документа
            logger.info(f"Начинаю обработку документа {file_path} с помощью {self.name}")
            result = await self.process(file_path, **kwargs)
            result.processing_time = asyncio.get_event_loop().time() - start_time

            logger.info(f"Обработка документа завершена за {result.processing_time:.2f}с")
            return result

        except ProcessingError as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"Ошибка обработки документа {file_path}: {e.message}")
            return DocumentResult.error_result(
                message=e.message,
                error_code=e.error_code,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.exception(f"Неожиданная ошибка при обработке документа {file_path}")
            return DocumentResult.error_result(
                message=f"Внутренняя ошибка: {str(e)}",
                error_code="INTERNAL_ERROR",
                processing_time=processing_time
            )

@dataclass
class DocumentInfo:
    """Информация о загруженном документе"""
    file_path: Path
    original_name: str
    size: int
    mime_type: str
    upload_time: datetime
    user_id: int

class DocumentStorage:
    """Класс для управления хранением документов"""

    def __init__(self, storage_path: Union[str, Path] = "data/documents"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()

    def get_user_storage_path(self, user_id: int) -> Path:
        """Получить путь к папке пользователя"""
        user_path = self.storage_path / str(user_id)
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path

    async def save_document(self, user_id: int, file_content: bytes, original_name: str, mime_type: str) -> DocumentInfo:
        """Сохранить документ пользователя"""
        user_path = self.get_user_storage_path(user_id)

        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(original_name).suffix
        safe_name = f"{timestamp}_{original_name.replace(' ', '_')}"
        file_path = user_path / safe_name

        # Сохраняем файл
        async with self._write_lock:
            with open(file_path, 'wb') as f:
                f.write(file_content)

        return DocumentInfo(
            file_path=file_path,
            original_name=original_name,
            size=len(file_content),
            mime_type=mime_type,
            upload_time=datetime.now(),
            user_id=user_id
        )

    def cleanup_old_files(self, user_id: int, max_age_hours: int = 24):
        """Очистка старых файлов пользователя"""
        user_path = self.get_user_storage_path(user_id)
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        for file_path in user_path.iterdir():
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    logger.info(f"Удален старый файл: {file_path}")
                except Exception as e:
                    logger.warning(f"Не удалось удалить файл {file_path}: {e}")