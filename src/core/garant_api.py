from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import httpx

logger = logging.getLogger(__name__)


class GarantAPIError(RuntimeError):
    """Raised when interaction with the Garant API fails."""


@dataclass(slots=True)
class GarantAncestor:
    entry: int
    title: str = ""


@dataclass(slots=True)
class GarantSnippet:
    entry: int
    relevance: float | None = None
    ancestors: list[GarantAncestor] = field(default_factory=list)

    def formatted_path(self, *, max_length: int = 140) -> str:
        titles = [ancestor.title.strip() for ancestor in self.ancestors if ancestor.title]
        path = " > ".join(titles)
        if len(path) > max_length:
            return path[: max_length - 3].rstrip() + "..."
        return path


@dataclass(slots=True)
class GarantDocument:
    topic: int
    name: str
    url: str | None = None

    def absolute_url(self, document_base_url: str | None) -> str | None:
        raw = (self.url or "").strip()
        if not raw:
            return None
        if raw.startswith("http://") or raw.startswith("https://"):
            return raw
        base = (document_base_url or "").strip().rstrip("/")
        if not base:
            return raw
        if raw.startswith("/"):
            return f"{base}{raw}"
        return f"{base}/{raw}"


@dataclass(slots=True)
class GarantSearchResult:
    document: GarantDocument
    snippets: list[GarantSnippet] = field(default_factory=list)


class GarantAPIClient:
    """Thin async wrapper around the Garant public API."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str | None = None,
        timeout: float = 15.0,
        default_kinds: Sequence[str] | None = None,
        result_limit: int = 3,
        snippet_limit: int = 2,
        document_base_url: str | None = None,
        use_query_language: bool = True,
        verify_ssl: bool = True,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token.strip() if token else None
        self._default_kinds = [kind for kind in (default_kinds or []) if kind]
        self._result_limit = max(1, result_limit)
        self._snippet_limit = max(0, snippet_limit)
        self._document_base_url = document_base_url
        self._use_query_language = use_query_language
        self._timeout = httpx.Timeout(timeout, connect=timeout, read=timeout)
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=headers,
            verify=verify_ssl,
        )

    @property
    def enabled(self) -> bool:
        return bool(self._base_url)

    @property
    def document_base_url(self) -> str | None:
        return self._document_base_url

    async def close(self) -> None:
        await self._client.aclose()

    async def search_documents(
        self,
        query: str,
        *,
        kinds: Sequence[str] | None = None,
        count: int | None = None,
        is_query: bool | None = None,
        sort: int = 0,
        sort_order: int = 0,
    ) -> list[GarantDocument]:
        if not self.enabled:
            return []
        text = (query or "").strip()
        if not text:
            return []

        payload: dict[str, Any] = {
            "text": text,
            "count": max(1, min(count or self._result_limit, 50)),
            "kind": list(kinds or self._default_kinds),
            "sort": sort,
            "sortOrder": sort_order,
        }
        if is_query if is_query is not None else self._use_query_language:
            payload["isQuery"] = True

        try:
            response = await self._client.post("/v1/search", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:  # noqa: PERF203
            raise GarantAPIError(f"Garant search request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise GarantAPIError(
                f"Garant search failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise GarantAPIError(f"Garant search request failed: {exc}") from exc
        except ValueError as exc:
            raise GarantAPIError(f"Invalid JSON from Garant search: {exc}") from exc

        documents_raw = data.get("documents") if isinstance(data, dict) else None
        documents: list[GarantDocument] = []
        if isinstance(documents_raw, Iterable):
            for item in documents_raw:
                doc = self._parse_document(item)
                if doc:
                    documents.append(doc)
        return documents[: payload["count"]]

    async def get_snippets(
        self,
        *,
        topic: int | str,
        text: str | None = None,
        correspondent: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[GarantSnippet]:
        if not self.enabled:
            return []
        payload: dict[str, Any] = {"topic": int(topic)}
        if text:
            payload["text"] = text.strip()
        if correspondent:
            payload["correspondent"] = correspondent

        if "text" in payload and "correspondent" in payload:
            payload.pop("correspondent")

        try:
            response = await self._client.post("/v1/snippets", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.TimeoutException as exc:  # noqa: PERF203
            raise GarantAPIError(f"Garant snippets request timed out: {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise GarantAPIError(
                f"Garant snippets failed with HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise GarantAPIError(f"Garant snippets request failed: {exc}") from exc
        except ValueError as exc:
            raise GarantAPIError(f"Invalid JSON from Garant snippets: {exc}") from exc

        snippets_raw = data.get("snippets") if isinstance(data, dict) else None
        snippets: list[GarantSnippet] = []
        if isinstance(snippets_raw, Iterable):
            for item in snippets_raw:
                snippet = self._parse_snippet(item)
                if snippet:
                    snippets.append(snippet)

        trimmed_limit = limit if limit is not None else self._snippet_limit
        if trimmed_limit:
            return snippets[: trimmed_limit]
        return snippets

    async def search_with_snippets(
        self,
        query: str,
        *,
        kinds: Sequence[str] | None = None,
        count: int | None = None,
        snippet_limit: int | None = None,
    ) -> list[GarantSearchResult]:
        documents = await self.search_documents(query, kinds=kinds, count=count)
        if not documents:
            return []

        results: list[GarantSearchResult] = []
        limit = snippet_limit if snippet_limit is not None else self._snippet_limit

        for document in documents:
            try:
                snippets = await self.get_snippets(topic=document.topic, text=query, limit=limit)
            except GarantAPIError as exc:
                logger.warning("Failed to load snippets for topic %s: %s", document.topic, exc)
                snippets = []
            results.append(GarantSearchResult(document=document, snippets=snippets))
            await asyncio.sleep(0)  # allow event loop to switch tasks
        return results

    def format_results(self, results: Sequence[GarantSearchResult]) -> str:
        return format_search_results(results, document_base_url=self._document_base_url)

    def _parse_document(self, item: Any) -> GarantDocument | None:
        if not isinstance(item, dict):
            return None
        topic_raw = item.get("topic")
        name = str(item.get("name") or "").strip()
        url = item.get("url")
        try:
            topic = int(topic_raw)
        except (TypeError, ValueError):
            return None
        if not name:
            name = f"Документ {topic}"
        return GarantDocument(topic=topic, name=name, url=url)

    def _parse_snippet(self, item: Any) -> GarantSnippet | None:
        if not isinstance(item, dict):
            return None
        entry_raw = item.get("entry")
        try:
            entry = int(entry_raw)
        except (TypeError, ValueError):
            return None
        relevance_raw = item.get("relevance")
        relevance: float | None = None
        if relevance_raw is not None:
            try:
                relevance = float(relevance_raw)
            except (TypeError, ValueError):
                relevance = None

        ancestors_raw = item.get("ancestors") or []
        ancestors: list[GarantAncestor] = []
        if isinstance(ancestors_raw, Iterable):
            for ancestor in ancestors_raw:
                if not isinstance(ancestor, dict):
                    continue
                ancestor_entry = ancestor.get("entry")
                title = str(ancestor.get("title") or "").strip()
                try:
                    ancestor_idx = int(ancestor_entry)
                except (TypeError, ValueError):
                    continue
                ancestors.append(GarantAncestor(entry=ancestor_idx, title=title))

        return GarantSnippet(entry=entry, relevance=relevance, ancestors=ancestors)


def format_search_results(
    results: Sequence[GarantSearchResult],
    *,
    document_base_url: str | None = None,
) -> str:
    if not results:
        return ""

    lines: list[str] = ["[ГАРАНТ] Результаты поиска судебной практики:"]

    for idx, result in enumerate(results, start=1):
        doc = result.document
        title = doc.name.strip() or f"Документ {doc.topic}"
        lines.append(f"{idx}. {title}")
        url = doc.absolute_url(document_base_url)
        if url:
            lines.append(f"    URL: {url}")
        lines.append(f"    topic: {doc.topic}")

        if result.snippets:
            lines.append("    Вхождения:")
            for snippet in result.snippets:
                path = snippet.formatted_path()
                relevance = (
                    f"{snippet.relevance:.2f}"
                    if isinstance(snippet.relevance, float)
                    else None
                )
                parts = [f"entry {snippet.entry}"]
                if relevance:
                    parts.append(f"rlv {relevance}")
                if path:
                    parts.append(path)
                lines.append("      - " + " | ".join(parts))
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    return "\n".join(lines)
