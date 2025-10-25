from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QuestionAttachment:
    """Container for user-supplied files attached to a legal question."""

    filename: str
    mime_type: str
    data: bytes

    @property
    def size(self) -> int:
        return len(self.data)
