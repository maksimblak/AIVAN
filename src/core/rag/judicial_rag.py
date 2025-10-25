from __future__ import annotations

import hashlib
import uuid
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from urllib.parse import urljoin

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
                self._embedding_service = EmbeddingService(
                    embedding_model or "text-embedding-3-large"
                )
                self._vector_store = self._build_vector_store(settings)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to initialize judicial RAG: %s", exc, exc_info=True)
                self._enabled = False

        self._last_fragments: list[PracticeFragment] = []

    @property
    def enabled(self) -> bool:
        return (
            self._enabled and self._embedding_service is not None and self._vector_store is not None
        )

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

    # ---------------------------
    # Garant/Sutyazhnik ingestion helpers
    # ---------------------------

    async def ingest_garant_fragments(
        self,
        fragments: Sequence[Any],
        *,
        max_items: int = 20,
        include_norms: bool = False,
    ) -> int:
        """Ingest pre-normalized Garant fragments (e.g. Excel builder output)."""
        if not self.enabled or not fragments or not max_items:
            return 0

        cap = max(0, int(max_items))
        items: list[tuple[object, str, dict[str, Any]]] = []
        seen_ids: set[str] = set()

        for fragment in list(fragments)[:cap]:
            match = getattr(fragment, "match", None)
            metadata = dict(getattr(match, "metadata", {}) or {})
            if not metadata:
                continue
            title = str(metadata.get("title") or metadata.get("name") or "").strip()
            if not title:
                continue
            source = str(metadata.get("source") or "").strip()
            if source == "sutyazhnik_norm" and not include_norms:
                continue

            topic = metadata.get("topic")
            entry = metadata.get("entry")
            url = str(metadata.get("url") or metadata.get("link") or "").strip()

            if isinstance(topic, int):
                point_id: object = int(topic)
            else:
                base = url or title or "garant"
                if entry is not None:
                    base += f"#entry:{entry}"
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"garant:{base}"))

            if str(point_id) in seen_ids:
                continue
            seen_ids.add(str(point_id))

            parts = [
                title,
                str(metadata.get("court") or "").strip(),
                str(metadata.get("decision_date") or metadata.get("date") or "").strip(),
                str(metadata.get("region") or "").strip(),
                str(metadata.get("summary") or "").strip(),
            ]
            text = " | ".join(part for part in parts if part).strip() or title

            payload = dict(metadata)
            payload.setdefault("doc_type", "norm" if source == "sutyazhnik_norm" else "court")
            payload.setdefault("source", source or "garant")
            payload["text"] = text
            items.append((point_id, text, payload))

        if not items:
            return 0
        return await self._upsert_items(items)

    async def ingest_sutyazhnik(
        self,
        results: Sequence[Any],
        query: str,
        *,
        include_norms: bool = False,
        max_items: int = 50,
        document_base_url: Optional[str] = "https://d.garant.ru",
    ) -> int:
        """Ingest Sutyazhnik references (courts + optional norms)."""
        if not self.enabled or not results or not max_items:
            return 0

        items: list[tuple[object, str, dict[str, Any]]] = []
        seen_ids: set[str] = set()
        court_cap = max(0, int(max_items))
        norm_cap = court_cap

        def _abs(url: str) -> str:
            if not url:
                return ""
            if url.startswith(("http://", "https://")):
                return url
            return urljoin(document_base_url or "", url)

        for block in results:
            kind = str(getattr(block, "kind", "") or "")
            norms_raw = [
                {
                    "topic": getattr(norm, "topic", None),
                    "name": getattr(norm, "name", "") or "",
                    "url": _abs(getattr(norm, "url", "") or ""),
                }
                for norm in (getattr(block, "norms", []) or [])
            ]

            for ref in (getattr(block, "courts", []) or [])[:court_cap]:
                topic = getattr(ref, "topic", None)
                name = str(getattr(ref, "name", "") or "").strip()
                url = _abs(getattr(ref, "url", "") or "")
                if topic is None or not name:
                    continue

                point_id: object = int(topic)
                if str(point_id) in seen_ids:
                    continue
                seen_ids.add(str(point_id))

                case_number, decision_date = self._parse_title_bits(name)
                payload: dict[str, Any] = {
                    "doc_type": "court",
                    "source": "garant_sutyazhnik",
                    "kind": kind,
                    "title": name,
                    "name": name,
                    "topic": int(topic),
                    "url": url,
                    "query": (query or "").strip(),
                    "case_number": case_number or "",
                    "decision_date": decision_date or "",
                    "sutyazhnik_norms": norms_raw,
                }
                text = " | ".join(
                    part
                    for part in [
                        name,
                        f"дело {case_number}" if case_number else "",
                        decision_date or "",
                    ]
                    if part
                )
                payload["text"] = text or name
                items.append((point_id, payload["text"], payload))

            if include_norms:
                for norm in (getattr(block, "norms", []) or [])[:norm_cap]:
                    topic = getattr(norm, "topic", None)
                    name = str(getattr(norm, "name", "") or "").strip()
                    url = _abs(getattr(norm, "url", "") or "")
                    if topic is None or not name:
                        continue
                    norm_id: object = str(uuid.uuid5(uuid.NAMESPACE_URL, f"garant:norm:{int(topic)}"))
                    if str(norm_id) in seen_ids:
                        continue
                    seen_ids.add(str(norm_id))
                    payload = {
                        "doc_type": "norm",
                        "source": "garant_sutyazhnik",
                        "kind": kind,
                        "title": name,
                        "name": name,
                        "topic": int(topic),
                        "url": url,
                        "query": (query or "").strip(),
                        "text": name,
                    }
                    items.append((norm_id, name, payload))

        if not items:
            return 0
        return await self._upsert_items(items)

    async def ingest_garant_search_results(
        self,
        results: Sequence[Any],
        query: str,
        *,
        max_items: int = 50,
        document_base_url: Optional[str] = "https://d.garant.ru",
    ) -> int:
        """Ingest Garant /v2/search documents (with snippets) directly into Qdrant."""
        if not self.enabled or not results or not max_items:
            return 0

        cap = max(0, int(max_items))
        items: list[tuple[object, str, dict[str, Any]]] = []
        seen_ids: set[str] = set()

        def _abs(url: str) -> str:
            if not url:
                return ""
            if url.startswith(("http://", "https://")):
                return url
            return urljoin(document_base_url or "", url)

        for result in list(results)[:cap]:
            document = getattr(result, "document", None)
            if not document:
                continue

            topic = getattr(document, "topic", None)
            name = str(getattr(document, "name", "") or "").strip()
            url = _abs(getattr(document, "url", "") or "")
            if topic is None or not name:
                continue

            point_id: object = int(topic)
            if str(point_id) in seen_ids:
                continue
            seen_ids.add(str(point_id))

            snippets = list(getattr(result, "snippets", []) or [])
            entry = getattr(snippets[0], "entry", None) if snippets else None
            case_number, decision_date = self._parse_title_bits(name)

            payload: dict[str, Any] = {
                "doc_type": "court",
                "source": "garant_search",
                "title": name,
                "name": name,
                "topic": int(topic),
                "url": url,
                "query": (query or "").strip(),
                "case_number": case_number or "",
                "decision_date": decision_date or "",
            }
            if entry is not None:
                payload["entry"] = int(entry)

            text = " | ".join(
                part
                for part in [
                    name,
                    f"entry {entry}" if entry is not None else "",
                    f"дело {case_number}" if case_number else "",
                    decision_date or "",
                ]
                if part
            )
            payload["text"] = text or name
            items.append((point_id, payload["text"], payload))

        if not items:
            return 0
        return await self._upsert_items(items)


    def _parse_title_bits(self, name: str) -> tuple[str | None, str | None]:
        if not name:
            return None, None
        case_match = re.search(r"по делу\s+N?\s*([A-Za-zА-Яа-я0-9\-/]+)", name, flags=re.IGNORECASE)
        date_match = re.search(r"от\s+(\d{1,2}\s+\S+\s+\d{4})", name, flags=re.IGNORECASE)
        return (
            case_match.group(1) if case_match else None,
            date_match.group(1) if date_match else None,
        )

    async def _upsert_items(self, items: list[tuple[object, str, dict[str, Any]]]) -> int:
        assert self._embedding_service is not None
        assert self._vector_store is not None

        texts = [text for _, text, _ in items]
        vectors = await self._embedding_service.embed(texts)
        if not vectors:
            return 0

        if len(vectors) != len(items):
            logger.warning(
                "Vector count mismatch during Garant ingestion (expected %s, got %s)",
                len(items),
                len(vectors),
            )
            limit = min(len(vectors), len(items))
            items = items[:limit]
            vectors = vectors[:limit]
            if not items or not vectors:
                return 0

        vector_size = len(vectors[0])
        await self._vector_store.ensure_collection(vector_size)
        await self._vector_store.upsert(
            vectors=vectors,
            payloads=[payload for _, _, payload in items],
            ids=[point_id for point_id, _, _ in items],
            batch_size=64,
        )
        return len(items)

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
