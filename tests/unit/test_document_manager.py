# mypy: ignore-errors
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from src.core.settings import AppSettings
from src.documents.base import DocumentInfo, DocumentResult
from src.documents.document_manager import DocumentManager


class DummyProcessor:
    def __init__(self, result: DocumentResult) -> None:
        self.result = result
        self.calls: list[tuple[Path, dict[str, object]]] = []

    async def safe_process(self, file_path: Path, **kwargs: object) -> DocumentResult:
        self.calls.append((Path(file_path), kwargs))
        return self.result


class DummyStorage:
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_document(
        self,
        *,
        user_id: int,
        file_content: bytes,
        original_name: str,
        mime_type: str,
    ) -> DocumentInfo:
        file_path = self.base_path / f"{uuid.uuid4().hex}.bin"
        file_path.write_bytes(file_content)
        return DocumentInfo(
            file_path=file_path,
            original_name=original_name,
            size=len(file_content),
            mime_type=mime_type,
            upload_time=datetime.now(),
            user_id=user_id,
            remote_path=None,
        )


def _make_settings(tmp_path: Path) -> AppSettings:
    env = {
        "TELEGRAM_BOT_TOKEN": "test-token",
        "OPENAI_API_KEY": "test-key",
        "DOCUMENTS_STORAGE_PATH": str(tmp_path),
    }
    return AppSettings.load(env)


@pytest.mark.asyncio
async def test_process_document_success(tmp_path: Path) -> None:
    manager = DocumentManager(openai_service=None, settings=_make_settings(tmp_path))
    manager.storage = DummyStorage(tmp_path)

    dummy_result = DocumentResult.success_result({"payload": "ok"})
    dummy_processor = DummyProcessor(dummy_result)
    manager.summarizer = dummy_processor  # type: ignore[assignment]
    manager._operations["summarize"]["processor"] = dummy_processor  # type: ignore[index]

    result = await manager.process_document(
        user_id=42,
        file_content=b"hello world",
        original_name="contract.txt",
        mime_type="text/plain",
        operation="summarize",
        detail_level="detailed",
    )

    assert result.success is True
    assert "document_info" in result.data
    assert result.data["document_info"]["original_name"] == "contract.txt"
    assert isinstance(result.data.get("exports", []), list)
    assert dummy_processor.calls, "processor should be invoked"
    stored_path, used_kwargs = dummy_processor.calls[0]
    assert stored_path.parent == tmp_path
    assert used_kwargs["detail_level"] == "detailed"


@pytest.mark.asyncio
async def test_process_document_unknown_operation(tmp_path: Path) -> None:
    manager = DocumentManager(openai_service=None, settings=_make_settings(tmp_path))
    manager.storage = DummyStorage(tmp_path)

    result = await manager.process_document(
        user_id=1,
        file_content=b"irrelevant",
        original_name="note.txt",
        mime_type="text/plain",
        operation="unknown",
    )

    assert result.success is False
    assert result.error_code == "UNKNOWN_OPERATION"
