import asyncio
from pathlib import Path

import pytest

from src.documents.document_manager import DocumentManager
from src.documents.base import DocumentResult, ProcessingError


class DummyProcessor:
    def __init__(self, result: DocumentResult) -> None:
        self.result = result
        self.calls: list[tuple[Path, dict[str, object]]] = []

    async def safe_process(self, file_path: Path, **kwargs: object) -> DocumentResult:
        self.calls.append((Path(file_path), kwargs))
        return self.result


@pytest.mark.asyncio
async def test_process_document_success(tmp_path: Path) -> None:
    manager = DocumentManager(openai_service=None, storage_path=tmp_path)
    dummy_result = DocumentResult.success_result({"payload": "ok"})
    dummy_processor = DummyProcessor(dummy_result)
    manager.summarizer = dummy_processor  # type: ignore[assignment]

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
    assert result.data["applied_options"]["detail_level"] == "detailed"
    assert dummy_processor.calls, "processor should be invoked"
    stored_path, used_kwargs = dummy_processor.calls[0]
    assert stored_path.exists()
    assert used_kwargs["detail_level"] == "detailed"


@pytest.mark.asyncio
async def test_process_document_unknown_operation(tmp_path: Path) -> None:
    manager = DocumentManager(openai_service=None, storage_path=tmp_path)
    with pytest.raises(ProcessingError):
        await manager.process_document(
            user_id=1,
            file_content=b"irrelevant",
            original_name="note.txt",
            mime_type="text/plain",
            operation="unknown",
        )
