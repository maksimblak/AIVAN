from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import re

from .storage_backends import ArtifactUploader  # fixed: relative import

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
    data: dict[str, Any]
    message: str
    processing_time: float
    timestamp: datetime
    error_code: str | None = None

    @classmethod
    def success_result(
        cls,
        data: dict[str, Any],
        message: str = "Обработка успешно завершена",
        processing_time: float = 0.0,
    ) -> "DocumentResult":
        """Создать успешный результат"""
        return cls(
            success=True,
            data=data,
            message=message,
            processing_time=processing_time,
            timestamp=datetime.now(),
        )

    @classmethod
    def error_result(
        cls, message: str, error_code: str = "PROCESSING_ERROR", processing_time: float = 0.0
    ) -> "DocumentResult":
        """Создать результат с ошибкой"""
        return cls(
            success=False,
            data={},
            message=message,
            processing_time=processing_time,
            timestamp=datetime.now(),
            error_code=error_code,
        )


class DocumentProcessor(ABC):
    """Базовый абстрактный класс для обработчиков документов"""

    def __init__(self, name: str, max_file_size: int = 50 * 1024 * 1024):  # 50MB по умолчанию
        self.name = name
        self.max_file_size = max_file_size
        self.supported_formats: list[str] = []

    @abstractmethod
    async def process(self, file_path: str | Path, **kwargs) -> DocumentResult:
        """Абстрактный метод для обработки документа"""
        raise NotImplementedError

    def validate_file(self, file_path: str | Path) -> tuple[bool, str]:
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
            return (
                False,
                f"Размер файла ({current_mb:.1f} МБ) превышает максимальный ({max_mb:.1f} МБ)",
            )

        # Проверка формата файла
        file_ext = path.suffix.lower()
        if self.supported_formats and file_ext not in self.supported_formats:
            return (
                False,
                f"Неподдерживаемый формат файла: {file_ext}. Поддерживаемые: {', '.join(self.supported_formats)}",
            )

        return True, "OK"

    async def safe_process(self, file_path: str | Path, **kwargs) -> DocumentResult:
        """Безопасная обработка документа с валидацией и обработкой ошибок"""
        try:
            loop = asyncio.get_running_loop()
            _now = loop.time  # monotonic
        except RuntimeError:
            loop = None
            _now = time.monotonic  # fallback if no loop yet

        start_time = _now()

        try:
            # Валидация файла
            is_valid, validation_message = self.validate_file(file_path)
            if not is_valid:
                processing_time = _now() - start_time
                return DocumentResult.error_result(
                    message=validation_message,
                    error_code="VALIDATION_ERROR",
                    processing_time=processing_time,
                )

            # Обработка документа
            logger.info("Начинаю обработку документа %s с помощью %s", file_path, self.name)
            result = await self.process(file_path, **kwargs)
            result.processing_time = _now() - start_time

            logger.info("Обработка документа завершена за %.2fс", result.processing_time)
            return result

        except ProcessingError as e:
            processing_time = _now() - start_time
            logger.error("Ошибка обработки документа %s: %s", file_path, e.message)
            return DocumentResult.error_result(
                message=e.message, error_code=e.error_code, processing_time=processing_time
            )

        except Exception as e:
            processing_time = _now() - start_time
            logger.exception("Неожиданная ошибка при обработке документа %s", file_path)
            return DocumentResult.error_result(
                message=f"Внутренняя ошибка: {str(e)}",
                error_code="INTERNAL_ERROR",
                processing_time=processing_time,
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
    remote_path: str | None = None


class DocumentStorage:
    """Manage on-disk document storage and optional remote uploads."""

    def __init__(
        self,
        storage_path: str | Path = "data/documents",
        *,
        max_user_quota_mb: int | None = None,
        cleanup_max_age_hours: int = 24,
        cleanup_interval_seconds: float = 3600.0,
        artifact_uploader: ArtifactUploader | None = None,
    ) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._write_lock = asyncio.Lock()
        self._max_quota_bytes = (
            int(max_user_quota_mb * 1024 * 1024) if max_user_quota_mb is not None else None
        )
        self._cleanup_max_age_hours = cleanup_max_age_hours
        self._cleanup_interval_seconds = cleanup_interval_seconds
        self._artifact_uploader = artifact_uploader

    @property
    def cleanup_interval_seconds(self) -> float:
        return self._cleanup_interval_seconds

    @property
    def cleanup_max_age_hours(self) -> int:
        return self._cleanup_max_age_hours

    def _sanitize_filename(self, filename: str) -> str:
        """Return a safe file name for storing on disk."""
        original = Path(filename).name
        ext = Path(original).suffix
        stem = original[: -len(ext)] if ext else original

        if len(stem) > 95:
            stem = stem[:95]

        safe_stem = re.sub(r"[^a-zA-Z0-9._-]", "_", stem)
        safe_stem = re.sub(r"_+", "_", safe_stem).strip("._")
        if not safe_stem:
            safe_stem = "document"

        safe_ext = ""
        if ext:
            base_ext = re.sub(r"[^a-zA-Z0-9]", "", ext)
            if base_ext:
                safe_ext = "." + base_ext if not base_ext.startswith(".") else base_ext
        return f"{safe_stem}{safe_ext}"

    async def save_document(
        self, user_id: int, file_content: bytes, original_name: str, mime_type: str
    ) -> DocumentInfo:
        """Persist a document to disk and optionally upload it to remote storage."""
        user_path = await asyncio.to_thread(self._ensure_user_path_sync, user_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{timestamp}_{self._sanitize_filename(original_name)}"
        file_path = user_path / safe_name
        file_size = len(file_content)

        async with self._write_lock:
            await self._enforce_user_quota(user_id, user_path, file_size)
            await asyncio.to_thread(self._write_file_sync, file_path, file_content)

        remote_path = await self._upload_artifact(file_path, mime_type)
        upload_time = datetime.now()

        return DocumentInfo(
            file_path=file_path,
            original_name=original_name,
            size=file_size,
            mime_type=mime_type,
            upload_time=upload_time,
            user_id=user_id,
            remote_path=remote_path,
        )

    async def cleanup_old_files(self, user_id: int, max_age_hours: int | None = None) -> int:
        user_path = self.storage_path / str(user_id)
        if not user_path.exists():
            return 0
        cutoff_time = self._compute_cutoff(max_age_hours)
        return await asyncio.to_thread(self._cleanup_user_directory_sync, user_path, cutoff_time)

    async def cleanup_all_users(self, max_age_hours: int | None = None) -> int:
        cutoff_time = self._compute_cutoff(max_age_hours)
        return await asyncio.to_thread(self._cleanup_all_users_sync, cutoff_time)

    async def get_user_usage(self, user_id: int) -> int:
        user_path = self.storage_path / str(user_id)
        return await self._get_directory_size(user_path)

    def _compute_cutoff(self, override_hours: int | None) -> float:
        hours = override_hours if override_hours is not None else self._cleanup_max_age_hours
        return datetime.now().timestamp() - (hours * 3600)

    def _ensure_user_path_sync(self, user_id: int) -> Path:
        user_path = self.storage_path / str(user_id)
        user_path.mkdir(parents=True, exist_ok=True)
        return user_path

    async def _get_directory_size(self, path: Path) -> int:
        if not path.exists():
            return 0
        return await asyncio.to_thread(self._calculate_directory_size, path)

    async def _enforce_user_quota(self, user_id: int, user_path: Path, incoming_size: int) -> None:
        if self._max_quota_bytes is None:
            return

        usage = await self._get_directory_size(user_path)
        if usage + incoming_size <= self._max_quota_bytes:
            self._log_quota_usage(user_id, usage + incoming_size)
            return

        if self._cleanup_max_age_hours > 0:
            cutoff = self._compute_cutoff(self._cleanup_max_age_hours)
            await asyncio.to_thread(self._cleanup_user_directory_sync, user_path, cutoff)
            usage = await self._get_directory_size(user_path)

        if usage + incoming_size > self._max_quota_bytes:
            raise ProcessingError(
                "Storage quota exceeded. Remove old files and try again.",
                "STORAGE_QUOTA_EXCEEDED",
            )

        self._log_quota_usage(user_id, usage + incoming_size)

    async def _upload_artifact(self, file_path: Path, mime_type: str) -> str | None:
        if self._artifact_uploader is None:
            return None
        try:
            return await self._artifact_uploader.upload(file_path, content_type=mime_type)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to upload document %s to remote storage: %s", file_path.name, exc)
            return None

    @staticmethod
    def _write_file_sync(file_path: Path, data: bytes) -> None:
        with open(file_path, "wb") as handle:
            handle.write(data)

    @staticmethod
    def _calculate_directory_size(path: Path) -> int:
        total = 0
        if not path.exists():
            return total
        for entry in path.iterdir():
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
        return total

    @staticmethod
    def _cleanup_user_directory_sync(user_path: Path, cutoff_time: float) -> int:
        removed = 0
        if not user_path.exists():
            return removed
        for file_path in user_path.iterdir():
            try:
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    logger.info("Removed stale document: %s", file_path)
                    removed += 1
            except OSError as exc:
                logger.warning("Failed to remove %s during cleanup: %s", file_path, exc)
        return removed

    def _cleanup_all_users_sync(self, cutoff_time: float) -> int:
        removed = 0
        for user_path in self.storage_path.iterdir():
            if user_path.is_dir():
                removed += self._cleanup_user_directory_sync(user_path, cutoff_time)
        return removed

    def _log_quota_usage(self, user_id: int, projected_usage: int) -> None:
        if self._max_quota_bytes in (None, 0):
            return
        ratio = projected_usage / self._max_quota_bytes
        percent = int(ratio * 100)
        if ratio >= 0.9:
            logger.warning(
                "User %s has used %s%% of the document storage quota",
                user_id,
                percent,
            )
        elif ratio >= 0.8:
            logger.info(
                "User %s has used %s%% of the document storage quota",
                user_id,
                percent,
            )
