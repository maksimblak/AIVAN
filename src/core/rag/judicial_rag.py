from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from src.core.settings import AppSettings

from .embedding_service import EmbeddingService
from .vector_store import QdrantVectorStore, VectorMatch

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class PracticeFragment:
    """Normalized slice of judicial practice returned by RAG."""

    match: VectorMatch
    header: str
    excerpt: str


class JudicialPracticeRAG:
    """Collects context snippets for judicial practice queries."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._enabled = settings.get_bool("RAG_ENABLED", False)
        self._top_k = max(1, settings.get_int("RAG_TOP_K", 6))
        self._snippet_char_limit = max(200, settings.get_int("RAG_SNIPPET_CHAR_LIMIT", 1200))
        self._context_char_limit = settings.get_int("RAG_CONTEXT_CHAR_LIMIT", 6500)

        embedding_model = settings.get_str("RAG_EMBEDDING_MODEL", "text-embedding-3-large")
        self._embedding_service: EmbeddingService | None = None
        self._vector_store: QdrantVectorStore | None = None

        if not self._enabled:
            logger.info("Judicial RAG disabled via settings")
        else:
            try:
                self._embedding_service = EmbeddingService(embedding_model or "text-embedding-3-large")
                self._vector_store = self._build_vector_store(settings)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to initialize judicial RAG: %s", exc, exc_info=True)
                self._enabled = False

        self._last_fragments: list[PracticeFragment] = []

    @property
    def enabled(self) -> bool:
        return self._enabled and self._embedding_service is not None and self._vector_store is not None

    @property
    def last_fragments(self) -> list[PracticeFragment]:
        return list(self._last_fragments)

    async def build_context(
        self,
        question: str,
        *,
        limit: int | None = None,
        filters: Any | None = None,
    ) -> tuple[str, list[PracticeFragment]]:
        if not self.enabled:
            return "", []

        normalized_question = question.strip()
        if not normalized_question:
            return "", []

        assert self._embedding_service is not None
        vectors = await self._embedding_service.embed([normalized_question])
        if not vectors:
            return "", []

        query_vector = vectors[0]
        assert self._vector_store is not None
        try:
            matches = await self._vector_store.search(
                query_vector,
                limit=limit or self._top_k,
                filters=filters,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Judicial RAG search failed: %s", exc, exc_info=True)
            return "", []

        fragments = self._normalise_matches(matches)
        if not fragments:
            return "", []

        context_blocks: list[str] = []
        total_chars = 0
        char_limit = self._context_char_limit or 0

        for idx, fragment in enumerate(fragments, start=1):
            header = fragment.header or f"Case {fragment.match.id}"
            block_lines = [f"[case {idx}] {header}".strip(), fragment.excerpt]
            block = "\n".join(line for line in block_lines if line)
            if char_limit and total_chars + len(block) > char_limit and context_blocks:
                break
            context_blocks.append(block)
            total_chars += len(block) + 2

        combined_context = "\n\n".join(context_blocks)
        self._last_fragments = fragments
        return combined_context, fragments

    async def close(self) -> None:
        if self._vector_store is not None:
            await self._vector_store.close()

    def _build_vector_store(self, settings: AppSettings) -> QdrantVectorStore:
        collection = settings.get_str("RAG_COLLECTION", "judicial_practice") or "judicial_practice"
        url = settings.get_str("RAG_QDRANT_URL")
        host = settings.get_str("RAG_QDRANT_HOST")
        api_key = settings.get_str("RAG_QDRANT_API_KEY")

        port_value = settings.get_str("RAG_QDRANT_PORT")
        port = int(port_value) if port_value else None

        prefer_grpc = settings.get_bool("RAG_QDRANT_GRPC", False)
        timeout_value = settings.get_str("RAG_QDRANT_TIMEOUT")
        timeout = float(timeout_value) if timeout_value else None

        score_raw = settings.get_str("RAG_SCORE_THRESHOLD")
        score_threshold = float(score_raw) if score_raw else None

        return QdrantVectorStore(
            collection=collection,
            url=url,
            host=host,
            port=port,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout,
            score_threshold=score_threshold,
        )

    def _normalise_matches(self, matches: Sequence[VectorMatch]) -> list[PracticeFragment]:
        fragments: list[PracticeFragment] = []
        for match in matches:
            excerpt = self._prepare_excerpt(match.text)
            if not excerpt:
                continue
            header = self._format_header(match.metadata)
            fragments.append(PracticeFragment(match=match, header=header, excerpt=excerpt))
        return fragments

    def _prepare_excerpt(self, raw: str) -> str:
        text = (raw or "").strip()
        if not text:
            return ""
        text = _WHITESPACE_RE.sub(" ", text)
        if len(text) > self._snippet_char_limit:
            text = text[: self._snippet_char_limit].rstrip()
            if not text.endswith("."):
                text += "..."
        return text

    def _format_header(self, metadata: dict[str, Any]) -> str:
        if not metadata:
            return ""
        parts: list[str] = []
        title = metadata.get("title") or metadata.get("name")
        case_number = metadata.get("case_number") or metadata.get("case")
        court = metadata.get("court")
        date = metadata.get("date") or metadata.get("decision_date")
        region = metadata.get("region")

        if title:
            parts.append(str(title))
        if case_number:
            parts.append(f"дело {case_number}")
        if court:
            parts.append(str(court))
        if date:
            parts.append(str(date))
        if region:
            parts.append(str(region))
        url = metadata.get("url") or metadata.get("link")
        if url:
            parts.append(str(url))
        return " • ".join(part.strip() for part in parts if part)
