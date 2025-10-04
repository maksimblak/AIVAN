from __future__ import annotations

import logging
from typing import Iterable, Sequence

from src.bot.openai_gateway import _make_async_client

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wrapper around OpenAI embeddings for reuse across RAG components."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def embed(self, texts: Sequence[str] | Iterable[str]) -> list[list[float]]:
        payload = list(texts)
        if not payload:
            return []

        try:
            async with await _make_async_client() as client:
                response = await client.embeddings.create(model=self.model, input=payload)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to obtain embeddings: %s", exc)
            raise

        vectors: list[list[float]] = []
        for item in getattr(response, "data", []):
            vector = getattr(item, "embedding", None)
            if isinstance(vector, list):
                vectors.append(vector)

        if len(vectors) != len(payload):
            logger.warning(
                "Embedding count mismatch (expected %s, got %s)", len(payload), len(vectors)
            )
        return vectors
